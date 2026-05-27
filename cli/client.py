"""
cli/client.py
-------------
Command-line client for the LankaMind inference pipeline.

Usage
-----
    python cli/client.py "Sri Lanka is a beautiful island"
    python cli/client.py "Once upon a time" --max-tokens 80 --workers 3

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


def run_client(
    prompt: str,
    max_new_tokens: int = 60,
    model: str = "gpt2",
    workers: int = 3,
    base_port: int = 5500,
    result_port: int = 5599,
) -> str:
    """
    Send *prompt* through the pipeline and return the generated continuation.

    Parameters
    ----------
    prompt         : The text the model will continue.
    max_new_tokens : Maximum number of tokens to generate.
    model          : HuggingFace tokeniser to use (must match the workers' model).
    workers        : Number of workers in the pipeline (informational only).
    base_port      : Port that Worker 0 listens on.
    result_port    : Port this client will bind to receive results.

    Returns
    -------
    str — the generated text (without the original prompt).
    """

    # ── Set up ZMQ ────────────────────────────────────────────────────────────
    ctx = zmq.Context()

    # PUSH → Worker 0 (worker binds, we connect)
    send_sock: zmq.Socket = ctx.socket(zmq.PUSH)
    send_sock.connect(f"tcp://localhost:{base_port}")

    # PULL ← last worker (we bind, worker connects)
    recv_sock: zmq.Socket = ctx.socket(zmq.PULL)
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

    args = parser.parse_args()

    run_client(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        model=args.model,
        workers=args.workers,
        base_port=args.base_port,
        result_port=args.result_port,
    )


if __name__ == "__main__":
    main()
