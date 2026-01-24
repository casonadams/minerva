<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { chatState, setLoadedModels, setSelectedModel } from '../stores';

	interface Props {
		onClose: () => void;
	}

	let { onClose }: Props = $props();

	let isDownloading = $state(false);
	let downloadProgress = $state<number | null>(null);

	const commonModels = [
		'meta-llama/Llama-2-7b',
		'mistralai/Mistral-7B',
		'microsoft/phi-2',
		'Qwen/Qwen-7B',
	];

	async function downloadModel(modelId: string) {
		isDownloading = true;
		downloadProgress = 0;

		try {
			await invoke('download_model', {
				model_id: modelId,
				local_dir: `./models/${modelId.split('/')[1]}`,
			});

			const models = [...$chatState.loadedModels, modelId];
			setLoadedModels(models);
			setSelectedModel(modelId);
			onClose();
		} catch (err) {
			alert(`Failed: ${err}`);
		} finally {
			isDownloading = false;
			downloadProgress = null;
		}
	}
</script>

<div class="panel">
	<h3>Download Model</h3>
	<p>Select a model from HuggingFace:</p>
	<div class="models">
		{#each commonModels as model}
			{@const downloaded = $chatState.loadedModels.includes(model)}
			<button
				class="model-btn"
				class:downloaded
				onclick={() => downloadModel(model)}
				disabled={isDownloading || downloaded}
			>
				{downloaded ? '✓' : ''}
				{model.split('/')[1]}
			</button>
		{/each}
	</div>
	{#if downloadProgress !== null}
		<div class="progress">
			<div class="bar" style="width: {downloadProgress}%"></div>
		</div>
		<p class="percent">{downloadProgress}%</p>
	{/if}
</div>

<style>
	.panel {
		position: absolute;
		top: 100%;
		right: 0;
		background: white;
		border-radius: 8px;
		padding: 1rem;
		margin-top: 0.5rem;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
		z-index: 1000;
		min-width: 280px;
	}

	h3 {
		margin: 0 0 0.5rem 0;
		font-size: 1rem;
		color: #333;
	}

	p {
		margin: 0 0 1rem 0;
		font-size: 0.9rem;
		color: #666;
	}

	.models {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin-bottom: 1rem;
	}

	.model-btn {
		padding: 0.75rem;
		background: #f5f5f5;
		border: 1px solid #ddd;
		border-radius: 6px;
		color: #333;
		cursor: pointer;
		text-align: left;
		transition: all 0.2s;
	}

	.model-btn:hover:not(:disabled) {
		background: #667eea;
		color: white;
		border-color: #667eea;
	}

	.model-btn:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.model-btn.downloaded {
		background: #e8f5e9;
		border-color: #4caf50;
		color: #2e7d32;
	}

	.progress {
		height: 8px;
		background: #f0f0f0;
		border-radius: 4px;
		overflow: hidden;
		margin-bottom: 0.5rem;
	}

	.bar {
		height: 100%;
		background: linear-gradient(90deg, #667eea, #764ba2);
		transition: width 0.3s;
	}

	.percent {
		margin: 0;
		font-size: 0.85rem;
		color: #666;
		text-align: center;
	}
</style>
