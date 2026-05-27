"""
cli/client.py
-------------
Command-line client for the LankaMind inference pipeline.

Usage
-----
    python cli/client.py "Sri Lanka is a beautiful island"
    python cli/client.py "Once upon a time" --max-tokens 80 --workers 3

    # With gateway (Phase 2):
    python cli/client.py "Once upon a time" --gateway tcp://localhost:5701

    # With encryption (Phase 3):
    python cli/client.py "Once upon a time" --encrypt --server-key <Z85-pubkey>

What it does
------------
  1. Tokenises the prompt using GPT-2's tokeniser.
  2. Sends the token IDs to Worker 0 via a ZMQ PUSH socket.
  3. Waits for Worker N-1 to send back the next predicted token ID.
  4. Appends that token to the sequence and repeats (generation loop).
  5. Prints each token as it arrives — streaming effect.
  6. Stops when it generates the EOS token or reaches --max-tokens.

Generation note
---------------
GPT-2 is a *text completion* model, not a chatbot.  It continues whatever
text you give it.  For best results, give it a sentence start like:
    "The history of Sri Lanka spans"
rather than a question like "What is Sri Lanka?" (that tends to produce
odd continuations).

For chat-style Q&A use a fine-tuned instruction model — Phase 5 will
support LLaMA-3-Instruct and similar models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

import torch
import zmq
from transformers import GPT2Tokenizer

from transport.zmq_transport import WorkerOutputSocket, deserialize_tensor, serialize_tensor


def _discover_chain(
    gateway_address: str,
    model: str,
    num_shards: int,
) -> list[dict] | None:
    """
    Ask the gateway for a worker chain.
    Returns list of worker dicts (ordered by shard_idx) or None on failure.
    """
    ctx = zmq.Context()
    sock: zmq.Socket = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5_000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(gateway_address)
    try:
        sock.send_json({"type": "get_chain", "model_name": model, "num_shards": num_shards})
        resp = sock.recv_json()
        if resp.get("status") == "ok":
            return resp["chain"]
        print(f"[Gateway] {resp.get('message', 'error')}", file=sys.stderr)
        return None
    except zmq.Again:
        print("[Gateway] Timeout — falling back to direct ports.", file=sys.stderr)
        return None
    finally:
        sock.close()
        ctx.term()


def run_client(
    prompt: str,
    max_new_tokens: int = 60,
    model: str = "gpt2",
    workers: int = 3,
    base_port: int = 5500,
    result_port: int = 5599,
    gateway_address: str | None = None,
    encrypt: bool = False,
    server_key: str | None = None,
) -> str:
    """
    Send *prompt* through the pipeline and return the generated continuation.

    Parameters
    ----------
    prompt           : The text the model will continue.
    max_new_tokens   : Maximum number of tokens to generate.
    model            : HuggingFace tokeniser to use (must match the workers).
    workers          : Number of workers (used when gateway is not available).
    base_port        : Port that Worker 0 listens on (used without gateway).
    result_port      : Port this client will bind to receive results.
    gateway_address  : If set, discover the chain from the gateway (Phase 2).
    encrypt          : If True, use ZMQ CURVE encryption (Phase 3).
    server_key       : Server's Z85 public key (required when encrypt=True).

    Returns
    -------
    str — the generated text (without the original prompt).
    """

    # ── Phase 2: discover chain from gateway (optional) ───────────────────────
    first_worker_port = base_port
    if gateway_address:
        chain = _discover_chain(gateway_address, model, workers)
        if chain:
            first_worker_port = chain[0]["port"]
            # Also pick up server public key from chain if not provided
            if encrypt and not server_key and chain[0].get("public_key"):
                server_key = chain[0]["public_key"]
            print(
                f"[Gateway] Chain: "
                + " → ".join(f"{w['host']}:{w['port']}" for w in chain),
                file=sys.stderr,
            )

    # ── Set up ZMQ ────────────────────────────────────────────────────────────
    ctx = zmq.Context()

    # Phase 3: encrypted sockets
    if encrypt and server_key:
        from transport.secure_transport import curve_available, make_secure_push, make_secure_pull
        if curve_available():
            print("[Encrypt] CURVE encryption enabled.", file=sys.stderr)
            send_sock = make_secure_push(ctx, f"tcp://localhost:{first_worker_port}", server_key)
            recv_sock = ctx.socket(zmq.PULL)  # result socket stays plain (client-bound server)
            recv_sock.bind(f"tcp://*:{result_port}")
        else:
            print("[Encrypt] CURVE not available — using plaintext.", file=sys.stderr)
            encrypt = False
            send_sock = ctx.socket(zmq.PUSH)
            send_sock.connect(f"tcp://localhost:{first_worker_port}")
            recv_sock = ctx.socket(zmq.PULL)
            recv_sock.bind(f"tcp://*:{result_port}")
    else:
        # PUSH → Worker 0 (worker binds, we connect)
        send_sock = ctx.socket(zmq.PUSH)
        send_sock.connect(f"tcp://localhost:{first_worker_port}")

        # PULL ← last worker (we bind, worker connects)
        recv_sock = ctx.socket(zmq.PULL)
        recv_sock.bind(f"tcp://*:{result_port}")

    recv_sock.RCVTIMEO = 60_000  # milliseconds — raise an error if no reply in 60 s

    # ── Tokenise ──────────────────────────────────────────────────────────────
    print(f"Loading tokeniser for '{model}' …", file=sys.stderr, flush=True)
    tokeniser = GPT2Tokenizer.from_pretrained(model)
    eos_id: int = tokeniser.eos_token_id

    input_ids: torch.Tensor = tokeniser.encode(prompt, return_tensors="pt")
    sequence: torch.Tensor = input_ids.clone()

    # ── Generation loop ───────────────────────────────────────────────────────
    print(f"\nPrompt : {prompt}")
    print(f"Output : ", end="", flush=True)

    generated_tokens: list[int] = []
    session_id = uuid.uuid4().hex[:8]   # short random ID to tag log lines

    try:
        for step in range(max_new_tokens):
            header = {
                "request_id": f"{session_id}-{step:04d}",
                "generation_step": step,
            }

            # Send the full current sequence to Worker 0
            send_sock.send_multipart([
                json.dumps(header).encode("utf-8"),
                serialize_tensor(sequence),
            ])

            # Wait for the last worker to return next_token_id
            try:
                result_bytes, _empty = recv_sock.recv_multipart()
            except zmq.Again:
                print(
                    "\n\n[Timeout: no response from workers within 60 s. "
                    "Are all workers running?]",
                    file=sys.stderr,
                )
                break

            result = json.loads(result_bytes.decode("utf-8"))
            next_token_id: int = result["next_token_id"]

            # Decode the token to text and stream it to stdout
            token_text = tokeniser.decode([next_token_id])
            print(token_text, end="", flush=True)

            generated_tokens.append(next_token_id)

            # Extend the sequence for the next generation step
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
            sequence = torch.cat([sequence, next_token_tensor], dim=-1)

            if next_token_id == eos_id:
                break   # model decided to stop

    finally:
        print()  # newline after streamed output
        send_sock.close()
        recv_sock.close()
        ctx.term()

    generated_text = tokeniser.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LankaMind CLI — send a prompt through the distributed pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        type=str,
        help='Text prompt, e.g. "Sri Lanka is a beautiful island"',
    )
    parser.add_argument("--max-tokens",  type=int, default=60,    help="Maximum tokens to generate")
    parser.add_argument("--model",       type=str, default="gpt2", help="Model name (must match workers)")
    parser.add_argument("--workers",     type=int, default=3,     help="Number of workers in the pipeline")
    parser.add_argument("--base-port",   type=int, default=5500,  help="Port of Worker 0")
    parser.add_argument("--result-port", type=int, default=5599,  help="Local port to receive results on")
    parser.add_argument(
        "--gateway",
        type=str,
        default=None,
        help="Gateway client address for chain discovery (e.g. tcp://localhost:5701)",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        default=False,
        help="Enable ZMQ CURVE end-to-end encryption (Phase 3)",
    )
    parser.add_argument(
        "--server-key",
        type=str,
        default=None,
        help="Worker 0's Z85 public key (required with --encrypt unless --gateway provides it)",
    )

    args = parser.parse_args()

    run_client(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        model=args.model,
        workers=args.workers,
        base_port=args.base_port,
        result_port=args.result_port,
        gateway_address=args.gateway,
        encrypt=args.encrypt,
        server_key=args.server_key,
    )


if __name__ == "__main__":
    main()
