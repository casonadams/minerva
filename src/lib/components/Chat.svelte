<script lang="ts">
	import { chat, models } from '../api/endpoints';
	import { chatState, addMessage, setLoading, type ChatMessage } from '../stores';
	import Messages from './Messages.svelte';
	import ModelSelector from './ModelSelector.svelte';
	import ChatInput from './ChatInput.svelte';

	let inputValue = $state('');
	let scrollContainer: HTMLDivElement;

	$effect.pre(() => {
		// Auto-load models
		loadModels();
	});

	async function loadModels() {
		try {
			const response = await models.list();
			const modelIds = response.data.map((m) => m.id);
			chatState.update((s) => ({ ...s, loadedModels: modelIds }));
		} catch (err) {
			console.error('Failed to load models:', err);
		}
	}

	async function sendMessage() {
		if (!inputValue.trim()) return;

		const state = $chatState;
		const userMsg: ChatMessage = {
			id: crypto.randomUUID(),
			role: 'user',
			content: inputValue,
			timestamp: Date.now(),
		};
		addMessage(userMsg);
		const message = inputValue;
		inputValue = '';
		scrollToBottom();

		setLoading(true);
		try {
			const response = await chat.completions({
				model: state.selectedModel,
				messages: [{ role: 'user', content: message }],
				max_tokens: 256,
			});

			const assistantMsg: ChatMessage = {
				id: crypto.randomUUID(),
				role: 'assistant',
				content: response.choices[0]?.message.content || 'No response',
				timestamp: Date.now(),
			};
			addMessage(assistantMsg);
			scrollToBottom();
		} catch (err) {
			addMessage({
				id: crypto.randomUUID(),
				role: 'assistant',
				content: `Error: ${err instanceof Error ? err.message : String(err)}`,
				timestamp: Date.now(),
			});
			scrollToBottom();
		} finally {
			setLoading(false);
		}
	}

	function scrollToBottom() {
		setTimeout(() => {
			if (scrollContainer) scrollContainer.scrollTop = scrollContainer.scrollHeight;
		}, 0);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
	}
</script>

<div class="chat-container">
	<div class="chat-header">
		<h1>Minerva Chat</h1>
		<ModelSelector />
	</div>

	<div class="chat-messages" bind:this={scrollContainer}>
		<Messages />
	</div>

	<ChatInput
		bind:inputValue
		onSend={sendMessage}
		onKeydown={handleKeydown}
		disabled={$chatState.isLoading}
	/>
</div>

<style>
	.chat-container {
		display: flex;
		flex-direction: column;
		height: 100vh;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
	}

	.chat-header {
		padding: 2rem;
		background: rgba(0, 0, 0, 0.3);
		color: white;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.chat-header h1 {
		margin: 0;
		font-size: 1.5rem;
		font-weight: 600;
	}

	.chat-messages {
		flex: 1;
		overflow-y: auto;
		padding: 2rem;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.chat-messages::-webkit-scrollbar {
		width: 8px;
	}

	.chat-messages::-webkit-scrollbar-track {
		background: rgba(255, 255, 255, 0.1);
	}

	.chat-messages::-webkit-scrollbar-thumb {
		background: rgba(255, 255, 255, 0.3);
		border-radius: 4px;
	}

	.chat-messages::-webkit-scrollbar-thumb:hover {
		background: rgba(255, 255, 255, 0.5);
	}
</style>
