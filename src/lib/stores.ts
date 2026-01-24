/// Shared Store State

import { writable } from 'svelte/store';

export interface ChatMessage {
	id: string;
	role: 'user' | 'assistant';
	content: string;
	timestamp: number;
	tokens?: number;
	tps?: number;
}

export interface ChatState {
	messages: ChatMessage[];
	selectedModel: string;
	isLoading: boolean;
	loadedModels: string[];
}

const initialState: ChatState = {
	messages: [],
	selectedModel: 'meta-llama/Llama-2-7b',
	isLoading: false,
	loadedModels: [],
};

export const chatState = writable<ChatState>(initialState);

export function addMessage(message: ChatMessage) {
	chatState.update((state) => ({
		...state,
		messages: [...state.messages, message],
	}));
}

export function setLoading(isLoading: boolean) {
	chatState.update((state) => ({ ...state, isLoading }));
}

export function setSelectedModel(model: string) {
	chatState.update((state) => ({ ...state, selectedModel: model }));
}

export function setLoadedModels(models: string[]) {
	chatState.update((state) => ({ ...state, loadedModels: models }));
}

export function clearMessages() {
	chatState.update((state) => ({ ...state, messages: [] }));
}
