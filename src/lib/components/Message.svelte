<script lang="ts">
	interface ChatMessage {
		id: string;
		role: 'user' | 'assistant';
		content: string;
		timestamp: number;
		tokens?: number;
		tps?: number;
	}

	let { message }: { message: ChatMessage } = $props();

	function formatTime(timestamp: number): string {
		const date = new Date(timestamp);
		return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}
</script>

<div class="message" class:user={message.role === 'user'} class:assistant={message.role === 'assistant'}>
	<div class="message-content">
		<p>{message.content}</p>
	</div>
	<div class="message-meta">
		<time>{formatTime(message.timestamp)}</time>
		{#if message.tokens}
			<span class="tokens">{message.tokens} tokens @ {message.tps?.toFixed(1)}t/s</span>
		{/if}
	</div>
</div>

<style>
	.message {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin-bottom: 0.5rem;
		animation: slideIn 0.3s ease-out;
	}

	@keyframes slideIn {
		from {
			opacity: 0;
			transform: translateY(10px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.message.user {
		align-items: flex-end;
	}

	.message.assistant {
		align-items: flex-start;
	}

	.message-content {
		padding: 1rem 1.25rem;
		border-radius: 12px;
		max-width: 80%;
		word-wrap: break-word;
		line-height: 1.5;
	}

	.message.user .message-content {
		background: #667eea;
		color: white;
		border-radius: 12px 0 12px 12px;
	}

	.message.assistant .message-content {
		background: rgba(255, 255, 255, 0.95);
		color: #333;
		border-radius: 0 12px 12px 12px;
	}

	.message-content p {
		margin: 0;
		font-size: 0.95rem;
	}

	.message-meta {
		display: flex;
		gap: 0.75rem;
		font-size: 0.75rem;
		color: rgba(255, 255, 255, 0.7);
		padding: 0 0.5rem;
	}

	time {
		opacity: 0.8;
	}

	.tokens {
		opacity: 0.6;
		font-style: italic;
	}
</style>
