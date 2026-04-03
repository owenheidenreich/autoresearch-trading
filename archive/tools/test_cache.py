#!/usr/bin/env python3
"""Quick test: verify Anthropic prompt caching works on this environment.

Makes 2 API calls with identical system prompt blocks (with cache_control markers).
The second call should show cache_read > 0.

Usage: ANTHROPIC_API_KEY=sk-... python3 tools/test_cache.py
"""
import os
import sys
import time

# Allow running from repo root or tools/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

def main():
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    # Use the same model as run_loop.py
    model = "claude-opus-4-20250514"
    print(f"Model: {model}")
    print(f"SDK version: {anthropic.__version__}")
    print()

    # Build system prompt with cache_control (same structure as run_loop.py)
    system_blocks = [
        {
            "type": "text",
            "text": "You are a test assistant. " * 200,  # ~1k tokens to meet cache minimum
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": "Additional context block. " * 100,
            "cache_control": {"type": "ephemeral"},
        },
    ]

    client = anthropic.Anthropic()

    for i in range(2):
        label = "CALL 1 (cache prime)" if i == 0 else "CALL 2 (should hit cache)"
        print(f"--- {label} ---")
        t0 = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=50,
            system=system_blocks,
            messages=[{"role": "user", "content": f"Say 'hello {i+1}' and nothing else."}],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        elapsed = time.time() - t0

        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        print(f"  Response: {response.content[0].text}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Input tokens:  {input_tokens}")
        print(f"  Cache create:  {cache_create}")
        print(f"  Cache read:    {cache_read}")
        print(f"  Output tokens: {output_tokens}")
        print()

        if i == 0:
            # Small delay to let cache propagate
            time.sleep(2)

    # Verdict
    if cache_read > 0:
        print("✓ CACHE IS WORKING — cache_read > 0 on second call")
        savings_pct = (cache_read / (cache_read + input_tokens)) * 100 if (cache_read + input_tokens) > 0 else 0
        print(f"  {cache_read} tokens served from cache ({savings_pct:.0f}% of system prompt)")
        return 0
    else:
        print("✗ CACHE NOT WORKING — cache_read = 0 on second call")
        print("  Check: SDK version >= 0.32.0, model supports caching, cache_control blocks present")
        return 1


if __name__ == "__main__":
    sys.exit(main())
