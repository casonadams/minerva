<script lang="ts">
	import { chatState } from '../stores';
	import Message from './Message.svelte';
</script>

{#each $chatState.messages as message (message.id)}
	<Message {message} />
{/each}

{#if $chatState.isLoading}
	<div class="loading">
		<div class="spinner"></div>
		<span>Generating response...</span>
	</div>
{/if}

<style>
	.loading {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		color: white;
		font-size: 0.9rem;
		animation: slideIn 0.3s ease-out;
	}

	.spinner {
		width: 20px;
		height: 20px;
		border: 3px solid rgba(255, 255, 255, 0.3);
		border-top-color: white;
		border-radius: 50%;
		animation: spin 1s linear infinite;
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

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
