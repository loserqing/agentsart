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
            outputFaceBlendshapes: true,
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

            if (face && faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
                const bs = faceResult.faceBlendshapes[0].categories;
                const bm = {};
                for (const c of bs) bm[c.categoryName] = c.score;
                this.sharedData.emotions = {
                    joy: Math.min(1, ((bm.mouthSmileLeft || 0) + (bm.mouthSmileRight || 0)) * 0.5),
                    tension: Math.min(1, ((bm.browDownLeft || 0) + (bm.browDownRight || 0)) * 0.5
                        + ((bm.mouthFrownLeft || 0) + (bm.mouthFrownRight || 0)) * 0.25),
                    surprise: Math.min(1, (bm.browInnerUp || 0) * 0.5 + (bm.jawOpen || 0) * 0.5),
                    calm: Math.min(1, ((bm.eyeBlinkLeft || 0) + (bm.eyeBlinkRight || 0)) * 0.5),
                };
            } else {
                this.sharedData.emotions = { joy: 0, tension: 0, surprise: 0, calm: 0 };
            }

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

    /**
     * 仅在人脸 landmark 有效时截取人脸区域；无人脸时不截图（不回落到全画面），避免空景/误检帧进入分析。
     * @returns {string|null} JPEG data URL 或 null
     */
    captureFrame() {
        const face = this.sharedData.face;
        if (!face || face.length < 10) return null;

        const maxRes = 512;
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;
        if (vw <= 0 || vh <= 0) return null;

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
        return this.captureCanvas.toDataURL('image/jpeg', 0.8);
    }
}
