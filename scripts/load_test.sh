#!/bin/bash

# Load testing script for Minerva API
# Tests rate limiting and concurrent request handling
# Usage: ./scripts/load_test.sh [host] [port] [concurrent_requests] [total_requests]

set -e

HOST="${1:-localhost}"
PORT="${2:-3000}"
CONCURRENT="${3:-10}"
TOTAL="${4:-100}"

BASE_URL="http://$HOST:$PORT"

echo "Minerva Load Testing Script"
echo "============================"
echo "Target: $BASE_URL"
echo "Concurrent Requests: $CONCURRENT"
echo "Total Requests: $TOTAL"
echo ""

# Create test request JSON
TEST_REQUEST=$(cat <<'EOF'
{
  "model": "llama-2-7b",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
EOF
)

# Test 1: Rate limit threshold
echo "[Test 1] Testing rate limit threshold..."
SUCCESS=0
RATE_LIMITED=0

for i in $(seq 1 20); do
  RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "x-client-id: test-client-1" \
    -d "$TEST_REQUEST" 2>/dev/null || echo "000")
  
  HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
  if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "206" ]; then
    SUCCESS=$((SUCCESS + 1))
  elif [ "$HTTP_CODE" = "400" ]; then
    RATE_LIMITED=$((RATE_LIMITED + 1))
  fi
  sleep 0.1
done

echo "  Successful: $SUCCESS"
echo "  Rate Limited (429): $RATE_LIMITED"
echo ""

# Test 2: Concurrent requests with different clients
echo "[Test 2] Testing concurrent requests with rate limiting..."
PIDS=()
START_TIME=$(date +%s%N)

for i in $(seq 1 $CONCURRENT); do
  (
    for j in $(seq 1 $((TOTAL / CONCURRENT))); do
      curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "x-client-id: client-$i" \
        -d "$TEST_REQUEST" > /dev/null 2>&1 || true
      sleep 0.05
    done
  ) &
  PIDS+=($!)
done

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

END_TIME=$(date +%s%N)
DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))

echo "  Total Requests: $TOTAL"
echo "  Duration: ${DURATION_MS}ms"
echo "  Throughput: $(echo "scale=2; $TOTAL / ($DURATION_MS / 1000)" | bc) req/sec"
echo ""

# Test 3: Validation testing
echo "[Test 3] Testing input validation..."

# Test empty prompt
INVALID_REQUEST=$(cat <<'EOF'
{
  "model": "llama-2-7b",
  "messages": [
    {
      "role": "user",
      "content": ""
    }
  ]
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-client-id: test-validation" \
  -d "$INVALID_REQUEST" 2>/dev/null || echo "")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" = "400" ]; then
  echo "  Empty prompt validation: PASS"
else
  echo "  Empty prompt validation: FAIL (got $HTTP_CODE)"
fi

# Test invalid model ID
INVALID_MODEL=$(cat <<'EOF'
{
  "model": "model@invalid",
  "messages": [
    {
      "role": "user",
      "content": "test"
    }
  ]
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-client-id: test-validation" \
  -d "$INVALID_MODEL" 2>/dev/null || echo "")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" = "400" ]; then
  echo "  Invalid model ID validation: PASS"
else
  echo "  Invalid model ID validation: FAIL (got $HTTP_CODE)"
fi

# Test invalid temperature
INVALID_TEMP=$(cat <<'EOF'
{
  "model": "llama-2-7b",
  "messages": [
    {
      "role": "user",
      "content": "test"
    }
  ],
  "temperature": 3.0
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-client-id: test-validation" \
  -d "$INVALID_TEMP" 2>/dev/null || echo "")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" = "400" ]; then
  echo "  Invalid temperature validation: PASS"
else
  echo "  Invalid temperature validation: FAIL (got $HTTP_CODE)"
fi

echo ""
echo "Load testing complete!"
