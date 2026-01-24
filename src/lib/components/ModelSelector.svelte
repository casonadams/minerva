<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { chatState, setSelectedModel, setLoadedModels } from '../stores';
	import DownloadPanel from './DownloadPanel.svelte';

	let showDownload = $state(false);

	function toggleDownload() {
		showDownload = !showDownload;
	}

	function handleModelChange(e: Event) {
		const target = e.target as HTMLSelectElement;
		setSelectedModel(target.value);
	}
</script>

<div class="selector">
	<label for="model">Model:</label>
	<select id="model" value={$chatState.selectedModel} onchange={handleModelChange}>
		<option disabled value="">Select a model...</option>
		{#each $chatState.loadedModels as model}
			<option value={model}>{model.split('/')[1] || model}</option>
		{/each}
	</select>
	<button onclick={toggleDownload} class="download-btn" title="Download new model">
		📥
	</button>

	{#if showDownload}
		<DownloadPanel onClose={() => (showDownload = false)} />
	{/if}
</div>

<style>
	.selector {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		position: relative;
	}

	label {
		color: white;
		font-weight: 600;
		font-size: 0.9rem;
		white-space: nowrap;
	}

	select {
		padding: 0.5rem 0.75rem;
		border: 1px solid rgba(255, 255, 255, 0.3);
		border-radius: 6px;
		background: rgba(255, 255, 255, 0.9);
		color: #333;
		font-size: 0.9rem;
		cursor: pointer;
		min-width: 150px;
	}

	select:hover {
		border-color: rgba(255, 255, 255, 0.5);
	}

	.download-btn {
		padding: 0.5rem 0.75rem;
		background: rgba(255, 255, 255, 0.2);
		border: 1px solid rgba(255, 255, 255, 0.3);
		color: white;
		border-radius: 6px;
		cursor: pointer;
		font-size: 1.1rem;
		transition: all 0.2s;
	}

	.download-btn:hover {
		background: rgba(255, 255, 255, 0.3);
		border-color: rgba(255, 255, 255, 0.5);
	}
</style>
