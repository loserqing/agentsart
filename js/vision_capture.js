/**
 * 摄像头捕捉 + MediaPipe Tasks API (FaceLandmarker + PoseLandmarker)
 * 替代旧版 @mediapipe/holistic；landmark 索引与旧版兼容。
 */
import {
    FaceLandmarker,
    PoseLandmarker,
    FilesetResolver,
} from '../assets/mediapipe-tasks/vision_bundle.mjs';

const DETECT_INTERVAL_MS = 33; // ~30 fps

export class VisionSystem {
    constructor(videoElement, sharedData) {
        this.video = videoElement;
        this.sharedData = sharedData;
        this.captureCanvas = document.createElement('canvas');
        this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: true });
        this._faceLandmarker = null;
        this._poseLandmarker = null;
        this._lastDetectTime = 0;
        this._rafId = null;
    }

    async init() {
        const vision = await FilesetResolver.forVisionTasks(
            './assets/mediapipe-tasks/wasm'
        );

        this._faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: './assets/mediapipe-tasks/models/face_landmarker.task',
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            minFaceDetectionConfidence: 0.55,
            minTrackingConfidence: 0.62,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
        });

        this._poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: './assets/mediapipe-tasks/models/pose_landmarker_lite.task',
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false,
        });
        this.video.srcObject = stream;
        await this.video.play();

        this._startDetectLoop();
    }

    _startDetectLoop() {
        const tick = () => {
            this._rafId = requestAnimationFrame(tick);
            const now = performance.now();
            if (now - this._lastDetectTime < DETECT_INTERVAL_MS) return;
            if (this.video.readyState < 2) return;
            this._lastDetectTime = now;

            const ts = now;

            const faceResult = this._faceLandmarker.detectForVideo(this.video, ts);
            const face = faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0
                ? faceResult.faceLandmarks[0]
                : null;
            this.sharedData.face = face;

            const poseResult = this._poseLandmarker.detectForVideo(this.video, ts);
            this.sharedData.pose = poseResult.landmarks && poseResult.landmarks.length > 0
                ? poseResult.landmarks[0]
                : null;

            if (this.video.videoWidth > 0) {
                this.sharedData.videoRes.x = this.video.videoWidth;
                this.sharedData.videoRes.y = this.video.videoHeight;
            }

            if (face) {
                this.sharedData.faceArea = Math.abs(face[10].y - face[152].y);
            } else {
                this.sharedData.faceArea = 0;
            }
        };
        this._rafId = requestAnimationFrame(tick);
    }

    captureFrame() {
        const maxRes = 512;

        if (this.sharedData.face && this.sharedData.face.length > 0) {
            const face = this.sharedData.face;
            const vw = this.video.videoWidth;
            const vh = this.video.videoHeight;

            let minX = 1, minY = 1, maxX = 0, maxY = 0;
            for (const pt of face) {
                if (pt.x < minX) minX = pt.x;
                if (pt.x > maxX) maxX = pt.x;
                if (pt.y < minY) minY = pt.y;
                if (pt.y > maxY) maxY = pt.y;
            }

            const cx = ((minX + maxX) / 2) * vw;
            const cy = ((minY + maxY) / 2) * vh;
            const w = (maxX - minX) * vw;
            const h = (maxY - minY) * vh;
            const size = Math.max(w, h) * 2.2;

            this.captureCanvas.width = Math.min(size, maxRes);
            this.captureCanvas.height = Math.min(size, maxRes);
            this.captureCtx.drawImage(this.video, cx - size / 2, cy - size / 2, size, size, 0, 0, this.captureCanvas.width, this.captureCanvas.height);
        } else {
            const scale = Math.min(1, maxRes / Math.max(this.video.videoWidth, this.video.videoHeight));
            this.captureCanvas.width = this.video.videoWidth * scale;
            this.captureCanvas.height = this.video.videoHeight * scale;
            this.captureCtx.drawImage(this.video, 0, 0, this.video.videoWidth, this.video.videoHeight, 0, 0, this.captureCanvas.width, this.captureCanvas.height);
        }
        return this.captureCanvas.toDataURL('image/jpeg', 0.8);
    }
}
