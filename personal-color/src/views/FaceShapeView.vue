<template>
	<div class="face-shape-analyzer">
		<header class="header">
			<h1>얼굴형 진단</h1>
			<p>사진을 업로드하여 나의 얼굴형을 확인하고 스타일 추천을 받아보세요</p>
		</header>

		<main class="main-content">
			<!-- 이미지 업로드 영역 -->
			<div class="upload-section">
				<div v-if="!previewImage" class="upload-options">
					<button
						class="upload-option-btn"
						:class="{ 'drag-over': isDragging }"
						@click="triggerFileInput"
						@drop.prevent="handleDrop"
						@dragover.prevent="isDragging = true"
						@dragleave.prevent="isDragging = false"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							width="48"
							height="48"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
						>
							<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
							<polyline points="17 8 12 3 7 8"></polyline>
							<line x1="12" y1="3" x2="12" y2="15"></line>
						</svg>
						<span>갤러리에서 선택</span>
						<span class="drag-hint">또는 드래그하여 업로드</span>
					</button>
					<button class="upload-option-btn" @click="openCamera">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							width="48"
							height="48"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
						>
							<path
								d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"
							></path>
							<circle cx="12" cy="13" r="4"></circle>
						</svg>
						<span>카메라로 촬영</span>
					</button>
				</div>

				<div v-if="showCamera" class="camera-container">
					<video
						ref="videoElement"
						autoplay
						playsinline
						class="camera-preview"
					></video>
					<div class="camera-controls">
						<button class="camera-btn capture-btn" @click="capturePhoto">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								width="32"
								height="32"
								viewBox="0 0 24 24"
								fill="white"
							>
								<circle cx="12" cy="12" r="10"></circle>
							</svg>
						</button>
						<button class="camera-btn cancel-btn" @click="closeCamera">
							취소
						</button>
					</div>
				</div>

				<div v-if="previewImage" class="preview-container">
					<img :src="previewImage" alt="Preview" class="preview-image" />
					<button class="remove-image" @click="removeImage">✕</button>
				</div>

				<input
					ref="fileInput"
					type="file"
					accept="image/*"
					@change="handleFileSelect"
					style="display: none"
				/>
				<button
					class="analyze-button"
					:disabled="!selectedFile || isAnalyzing"
					@click="analyzeFaceShape"
				>
					{{ isAnalyzing ? "분석 중..." : "얼굴형 진단하기" }}
				</button>
			</div>

			<!-- 분석 결과 영역 -->
			<div v-if="analysisResult" class="result-section">
				<div class="result-header">
					<h2>진단 결과</h2>
					<div class="shape-badge">{{ analysisResult.face_shape }}</div>
				</div>

				<div class="result-content">
					<div class="confidence-meter">
						<label>신뢰도</label>
						<div class="meter">
							<div
								class="meter-fill"
								:style="{ width: analysisResult.confidence + '%' }"
							></div>
						</div>
						<span>{{ analysisResult.confidence }}%</span>
					</div>

					<div class="description">
						<h3>얼굴형 설명</h3>
						<p>{{ analysisResult.description }}</p>
					</div>

					<!-- 레이더 차트 -->
					<div class="chart-section">
						<h3>얼굴형 확률 분포</h3>
						<div class="chart-container">
							<v-chart :option="chartOption" autoresize />
						</div>
					</div>

					<!-- 추천 헤어스타일 -->
					<div class="recommendations">
						<h3>추천 헤어스타일</h3>
						<ul class="recommendation-list">
							<li
								v-for="(style, index) in analysisResult.recommended_hairstyles"
								:key="'hair-' + index"
							>
								{{ style }}
							</li>
						</ul>
					</div>

					<!-- 추천 안경테 -->
					<div class="recommendations">
						<h3>추천 안경테</h3>
						<ul class="recommendation-list">
							<li
								v-for="(glasses, index) in analysisResult.recommended_glasses"
								:key="'glasses-' + index"
							>
								{{ glasses }}
							</li>
						</ul>
					</div>
				</div>
			</div>

			<!-- 에러 메시지 -->
			<div v-if="errorMessage" class="error-message">
				{{ errorMessage }}
			</div>
		</main>
	</div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { RadarChart } from "echarts/charts";
import {
	TitleComponent,
	TooltipComponent,
	LegendComponent,
} from "echarts/components";

// ECharts 컴포넌트 등록
use([
	CanvasRenderer,
	RadarChart,
	TitleComponent,
	TooltipComponent,
	LegendComponent,
]);

interface FaceShapeResult {
	face_shape: string;
	confidence: number;
	description: string;
	recommended_hairstyles: string[];
	recommended_glasses: string[];
	probabilities: Record<string, number>;
}

// 얼굴형 타입 고정 순서 (레이더 차트 축 일관성 유지)
const FACE_SHAPE_ORDER = ["긴형", "하트형", "사각형", "둥근형", "계란형"];

const API_URL = import.meta.env.PROD ? "" : "http://localhost:8000";

const fileInput = ref<HTMLInputElement | null>(null);
const videoElement = ref<HTMLVideoElement | null>(null);
const selectedFile = ref<File | null>(null);
const previewImage = ref<string | null>(null);
const isDragging = ref(false);
const isAnalyzing = ref(false);
const showCamera = ref(false);
const mediaStream = ref<MediaStream | null>(null);
const analysisResult = ref<FaceShapeResult | null>(null);
const errorMessage = ref<string | null>(null);

// ECharts 레이더 차트 옵션
const chartOption = computed(() => {
	if (!analysisResult.value) return {};

	const probabilities = analysisResult.value.probabilities;
	// 고정된 순서로 레이블과 값 매핑 (축 위치 일관성 유지)
	const labels = FACE_SHAPE_ORDER;
	const values = FACE_SHAPE_ORDER.map((shape) => probabilities[shape] || 0);

	return {
		tooltip: {
			trigger: "item",
			formatter: (params: any) => {
				return `${params.name}: ${params.value}%`;
			},
		},
		radar: {
			indicator: labels.map((label) => ({
				name: label,
				max: 100,
			})),
			shape: "polygon",
			splitNumber: 5,
			axisName: {
				color: "#333",
				fontSize: 14,
			},
			splitLine: {
				lineStyle: {
					color: ["#e5e7eb", "#e5e7eb", "#e5e7eb", "#e5e7eb", "#e5e7eb"],
				},
			},
			splitArea: {
				show: true,
				areaStyle: {
					color: ["rgba(74, 144, 226, 0.05)", "rgba(74, 144, 226, 0.1)"],
				},
			},
		},
		series: [
			{
				name: "얼굴형 확률",
				type: "radar",
				data: [
					{
						value: values,
						name: "확률 분포",
						areaStyle: {
							color: "rgba(74, 144, 226, 0.3)",
						},
						lineStyle: {
							color: "rgba(74, 144, 226, 1)",
							width: 2,
						},
						itemStyle: {
							color: "rgba(74, 144, 226, 1)",
						},
					},
				],
			},
		],
	};
});

const triggerFileInput = () => {
	fileInput.value?.click();
};

const handleFileSelect = (event: Event) => {
	const target = event.target as HTMLInputElement;
	const file = target.files?.[0];
	if (file) {
		processFile(file);
	}
};

const handleDrop = (event: DragEvent) => {
	isDragging.value = false;
	const file = event.dataTransfer?.files[0];
	if (file && file.type.startsWith("image/")) {
		processFile(file);
	}
};

const processFile = (file: File) => {
	if (file.size > 10 * 1024 * 1024) {
		errorMessage.value = "파일 크기는 10MB 이하여야 합니다.";
		return;
	}

	selectedFile.value = file;
	errorMessage.value = null;
	analysisResult.value = null;

	const reader = new FileReader();
	reader.onload = (e) => {
		previewImage.value = e.target?.result as string;
	};
	reader.readAsDataURL(file);
};

const removeImage = () => {
	selectedFile.value = null;
	previewImage.value = null;
	analysisResult.value = null;
	if (fileInput.value) {
		fileInput.value.value = "";
	}
};

const openCamera = async () => {
	try {
		showCamera.value = true;
		errorMessage.value = null;

		const stream = await navigator.mediaDevices.getUserMedia({
			video: { facingMode: "user" },
		});

		mediaStream.value = stream;

		if (videoElement.value) {
			videoElement.value.srcObject = stream;
		}
	} catch (error) {
		errorMessage.value = "카메라에 접근할 수 없습니다. 권한을 확인해주세요.";
		showCamera.value = false;
		console.error("Camera error:", error);
	}
};

const closeCamera = () => {
	if (mediaStream.value) {
		mediaStream.value.getTracks().forEach((track) => track.stop());
		mediaStream.value = null;
	}
	showCamera.value = false;
};

const capturePhoto = () => {
	if (!videoElement.value) return;

	const canvas = document.createElement("canvas");
	canvas.width = videoElement.value.videoWidth;
	canvas.height = videoElement.value.videoHeight;

	const context = canvas.getContext("2d");
	if (!context) return;

	context.drawImage(videoElement.value, 0, 0);

	canvas.toBlob(
		(blob) => {
			if (!blob) return;

			const file = new File([blob], "camera-photo.jpg", { type: "image/jpeg" });
			processFile(file);
			closeCamera();
		},
		"image/jpeg",
		0.9
	);
};

const analyzeFaceShape = async () => {
	if (!selectedFile.value) return;

	isAnalyzing.value = true;
	errorMessage.value = null;

	try {
		const formData = new FormData();
		formData.append("image", selectedFile.value);

		const response = await fetch(`${API_URL}/api/analyze/face-shape`, {
			method: "POST",
			body: formData,
		});

		if (!response.ok) {
			throw new Error("분석 요청에 실패했습니다.");
		}

		const result = await response.json();
		analysisResult.value = result;
	} catch (error) {
		errorMessage.value =
			error instanceof Error ? error.message : "분석 중 오류가 발생했습니다.";
		console.error("Analysis error:", error);
	} finally {
		isAnalyzing.value = false;
	}
};
</script>

<style scoped>
.face-shape-analyzer {
	max-width: 1200px;
	margin: 0 auto;
	padding: 2rem;
}

.header {
	text-align: center;
	margin-bottom: 3rem;
}

.header h1 {
	font-size: 2.5rem;
	margin-bottom: 0.5rem;
	color: #000000;
}

.header p {
	color: #333333;
	font-size: 1.1rem;
}

.main-content {
	display: grid;
	gap: 2rem;
}

.upload-section {
	background: white;
	border-radius: 12px;
	padding: 2rem;
	box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.upload-options {
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 1rem;
	margin-bottom: 1.5rem;
}

.upload-option-btn {
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 1rem;
	padding: 2rem 1rem;
	background: white;
	border: 2px solid #e0e0e0;
	border-radius: 12px;
	cursor: pointer;
	transition: all 0.3s ease;
	color: #000000;
	font-size: 1rem;
	font-weight: 500;
}

.upload-option-btn:hover {
	border-color: #4a90e2;
	background: #e3f2fd;
	transform: translateY(-2px);
	box-shadow: 0 4px 12px rgba(74, 144, 226, 0.2);
}

.upload-option-btn svg {
	color: #4a90e2;
}

.upload-option-btn.drag-over {
	border-color: #4a90e2;
	background: #e3f2fd;
	border-style: dashed;
}

.drag-hint {
	font-size: 0.85rem;
	color: #999;
	font-weight: 400;
	margin-top: 0.25rem;
}

.camera-container {
	position: relative;
	margin-bottom: 1.5rem;
	border-radius: 12px;
	overflow: hidden;
	background: #000000;
}

.camera-preview {
	width: 100%;
	max-height: 500px;
	object-fit: contain;
	display: block;
}

.camera-controls {
	position: absolute;
	bottom: 0;
	left: 0;
	right: 0;
	display: flex;
	justify-content: center;
	align-items: center;
	gap: 1rem;
	padding: 1.5rem;
	background: linear-gradient(to top, rgba(0, 0, 0, 0.7), transparent);
}

.camera-btn {
	border: none;
	border-radius: 50%;
	cursor: pointer;
	transition: all 0.3s ease;
	display: flex;
	align-items: center;
	justify-content: center;
}

.capture-btn {
	width: 70px;
	height: 70px;
	background: white;
	padding: 0;
}

.capture-btn:hover {
	transform: scale(1.1);
	box-shadow: 0 4px 12px rgba(255, 255, 255, 0.5);
}

.cancel-btn {
	padding: 0.75rem 1.5rem;
	background: rgba(255, 255, 255, 0.2);
	color: white;
	font-size: 1rem;
	font-weight: 500;
	border-radius: 8px;
}

.cancel-btn:hover {
	background: rgba(255, 255, 255, 0.3);
}

.preview-container {
	position: relative;
	max-width: 500px;
	margin: 0 auto;
	margin-bottom: 1.5rem;
}

.preview-image {
	width: 100%;
	border-radius: 8px;
	box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.remove-image {
	position: absolute;
	top: 10px;
	right: 10px;
	background: rgba(0, 0, 0, 0.7);
	color: white;
	border: none;
	border-radius: 50%;
	width: 32px;
	height: 32px;
	cursor: pointer;
	font-size: 1.2rem;
	transition: background 0.2s;
}

.remove-image:hover {
	background: rgba(0, 0, 0, 0.9);
}

.analyze-button {
	width: 100%;
	padding: 1rem 2rem;
	font-size: 1.1rem;
	font-weight: 600;
	color: white;
	background: #4a90e2;
	border: none;
	border-radius: 8px;
	cursor: pointer;
	transition: all 0.3s ease;
}

.analyze-button:hover:not(:disabled) {
	transform: translateY(-2px);
	background: #357abd;
	box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
}

.analyze-button:disabled {
	opacity: 0.5;
	cursor: not-allowed;
	background: #9e9e9e;
}

.result-section {
	background: white;
	border-radius: 12px;
	padding: 2rem;
	box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.result-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 2rem;
	padding-bottom: 1rem;
	border-bottom: 2px solid #e2e8f0;
}

.result-header h2 {
	font-size: 1.8rem;
	color: #000000;
}

.shape-badge {
	padding: 0.5rem 1.5rem;
	border-radius: 20px;
	font-weight: 600;
	font-size: 1.1rem;
	background: #4a90e2;
	color: white;
}

.confidence-meter {
	margin-bottom: 2rem;
}

.confidence-meter label {
	display: block;
	margin-bottom: 0.5rem;
	font-weight: 600;
	color: #000000;
}

.meter {
	height: 24px;
	background: #e2e8f0;
	border-radius: 12px;
	overflow: hidden;
	margin-bottom: 0.5rem;
}

.meter-fill {
	height: 100%;
	background: #4a90e2;
	transition: width 0.5s ease;
}

.description {
	margin-bottom: 2rem;
}

.description h3 {
	margin-bottom: 1rem;
	color: #000000;
}

.description p {
	line-height: 1.6;
	color: #333333;
}

.chart-section {
	margin-bottom: 2rem;
	padding: 1.5rem;
	background: #f8f9fa;
	border-radius: 12px;
}

.chart-section h3 {
	margin-bottom: 1.5rem;
	color: #000000;
	text-align: center;
}

.chart-container {
	max-width: 500px;
	height: 400px;
	margin: 0 auto;
	padding: 1rem;
}

.recommendations {
	margin-bottom: 2rem;
}

.recommendations:last-child {
	margin-bottom: 0;
}

.recommendations h3 {
	margin-bottom: 1rem;
	color: #000000;
}

.recommendation-list {
	list-style: none;
	padding: 0;
	margin: 0;
	display: grid;
	gap: 0.75rem;
}

.recommendation-list li {
	padding: 1rem 1.25rem;
	background: #f7fafc;
	border-left: 4px solid #4a90e2;
	border-radius: 8px;
	color: #333;
	transition: all 0.2s;
}

.recommendation-list li:hover {
	background: #e3f2fd;
	transform: translateX(5px);
}

.error-message {
	background: #fee;
	color: #c33;
	padding: 1rem;
	border-radius: 8px;
	margin-top: 1rem;
	text-align: center;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
	.face-shape-analyzer {
		padding: 1rem;
	}

	.header h1 {
		font-size: 1.8rem;
	}

	.header p {
		font-size: 1rem;
	}

	.upload-section {
		padding: 1.5rem;
	}

	.upload-options {
		grid-template-columns: 1fr;
	}

	.upload-option-btn:hover {
		border-color: #e0e0e0;
		background: white;
		transform: none;
		box-shadow: none;
	}

	.analyze-button:hover:not(:disabled) {
		transform: none;
		background: #4a90e2;
		box-shadow: none;
	}

	.result-section {
		padding: 1.5rem;
	}

	.result-header {
		flex-direction: column;
		gap: 1rem;
		align-items: flex-start;
	}

	.chart-container {
		padding: 0.5rem;
	}

	.recommendation-list li:hover {
		background: #f7fafc;
		transform: none;
	}
}
</style>
