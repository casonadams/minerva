<script lang="ts">
	import { chatState } from '../stores';

	interface Props {
		inputValue: string;
		onSend: () => void;
		onKeydown: (e: KeyboardEvent) => void;
		disabled?: boolean;
	}

	let { inputValue = $bindable(), onSend, onKeydown, disabled = false }: Props = $props();
</script>

<div class="input-area">
	<textarea
		bind:value={inputValue}
		placeholder="Type your message... (Shift+Enter for newline)"
		disabled={disabled || $chatState.isLoading}
		onkeydown={onKeydown}
		rows="3"
	></textarea>
	<button onclick={onSend} disabled={disabled || !inputValue.trim() || $chatState.isLoading}>
		{$chatState.isLoading ? 'Generating...' : 'Send'}
	</button>
</div>

<style>
	.input-area {
		padding: 2rem;
		background: rgba(0, 0, 0, 0.3);
		display: flex;
		gap: 1rem;
		border-top: 1px solid rgba(255, 255, 255, 0.2);
	}

	textarea {
		flex: 1;
		padding: 1rem;
		border: 1px solid rgba(255, 255, 255, 0.3);
		border-radius: 8px;
		background: rgba(255, 255, 255, 0.9);
		color: #333;
		font-family: inherit;
		font-size: 1rem;
		resize: vertical;
		min-height: 60px;
		max-height: 120px;
	}

	textarea:disabled {
		opacity: 0.6;
	}

	button {
		padding: 1rem 2rem;
		background: #fff;
		color: #667eea;
		border: none;
		border-radius: 8px;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.2s;
		white-space: nowrap;
	}

	button:hover:not(:disabled) {
		transform: translateY(-2px);
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
	}

	button:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
</style>
