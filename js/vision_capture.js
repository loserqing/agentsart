/**
 * 摄像头捕捉 + MediaPipe Tasks API (FaceLandmarker + PoseLandmarker)
 * 替代旧版 @mediapipe/holistic；landmark 索引与旧版兼容。
 */
import {
    FaceLandmarker,
    PoseLandmarker,
    HandLandmarker,
    FilesetResolver,
} from '../assets/mediapipe-tasks/vision_bundle.mjs';

const DETECT_INTERVAL_MS = 33; // ~30 fps
const HAND_DETECT_INTERVAL_MS = 66; // 手势只为触发，降到 ~15fps 节省算力

/**
 * MediaPipe HandLandmarker 21 点：0 腕，4/8/12/16/20 为拇/食/中/无名/小指尖，
 * 5/9/13/17 为各指根 MCP，6/10/14/18 为 PIP。
 * OK 手势判据：
 *   1) 拇指尖 ↔ 食指尖 距离 < 手宽的 0.18（捏合成圈）
 *   2) 中/无名/小指尖 y < 其 PIP y（指头伸直向上，图像坐标 y 向下）
 *   3) 不是完全握拳 —— 三指尖到腕的距离大于指根到腕距离的 1.05
 */
function classifyHandGesture(hand) {
    if (!hand || hand.length < 21) return null;
    const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const wrist = hand[0];
    const indexMcp = hand[5];
    const pinkyMcp = hand[17];
    const handWidth = dist(indexMcp, pinkyMcp);
    if (handWidth < 1e-4) return null;

    const thumbTip = hand[4];
    const indexTip = hand[8];
    const pinch = dist(thumbTip, indexTip) / handWidth;
    if (pinch > 0.55) return null;

    const middleExtended = hand[12].y < hand[10].y - 0.01;
    const ringExtended = hand[16].y < hand[14].y - 0.01;
    const pinkyExtended = hand[20].y < hand[18].y - 0.01;
    const extendedCount = [middleExtended, ringExtended, pinkyExtended].filter(Boolean).length;
    if (extendedCount < 2) return null;

    const midTipToWrist = dist(hand[12], wrist);
    const midMcpToWrist = dist(hand[9], wrist);
    if (midTipToWrist < midMcpToWrist * 1.05) return null;

    return 'ok';
}

export class VisionSystem {
    constructor(videoElement, sharedData) {
        this.video = videoElement;
        this.sharedData = sharedData;
        this.captureCanvas = document.createElement('canvas');
        this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: true });
        this._faceLandmarker = null;
        this._poseLandmarker = null;
        this._handLandmarker = null;
        this._lastDetectTime = 0;
        this._lastHandDetectTime = 0;
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
            // 略提高阈值，减少手掌/织物等误检为人脸（误检会导致服务端长期不触发 clear）
            minFaceDetectionConfidence: 0.62,
            minTrackingConfidence: 0.65,
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

            try {
            this._handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: './assets/mediapipe-tasks/models/hand_landmarker.task',
                    delegate: 'GPU',
                },
                runningMode: 'VIDEO',
                numHands: 2,
                minHandDetectionConfidence: 0.6,
                minHandPresenceConfidence: 0.6,
                minTrackingConfidence: 0.5,
            });
        } catch (e) {
            console.warn('[VisionSystem] HandLandmarker init failed, gesture trigger disabled:', e);
            this._handLandmarker = null;
        }

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

            if (this._handLandmarker && now - this._lastHandDetectTime >= HAND_DETECT_INTERVAL_MS) {
                this._lastHandDetectTime = now;
                const handResult = this._handLandmarker.detectForVideo(this.video, ts);
                const list = handResult.landmarks && handResult.landmarks.length > 0
                    ? handResult.landmarks
                    : null;
                let gesture = null;
                let primaryHand = null;
                if (list) {
                    for (const h of list) {
                        if (classifyHandGesture(h) === 'ok') {
                            gesture = 'ok';
                            primaryHand = h;
                            break;
                        }
                    }
                    if (!primaryHand) primaryHand = list[0];
                }
                this.sharedData.hand = primaryHand;
                this.sharedData.handGesture = gesture;
            }
        };
        this._rafId = requestAnimationFrame(tick);
    }

    /**
     * 仅在人脸 landmark 有效时截取人脸区域；无人脸时不截图（不回落到全画面），避免空景/误检帧进入分析。
     * @returns {string|null} JPEG data URL 或 null
     */
    /**
     * 误检脸（如手掌）常见：框极扁/极长或过小，与真人脸比例差异大。
     */
    _faceLandmarksBBoxLooksPlausible(face) {
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        for (const pt of face) {
            if (pt.x < minX) minX = pt.x;
            if (pt.x > maxX) maxX = pt.x;
            if (pt.y < minY) minY = pt.y;
            if (pt.y > maxY) maxY = pt.y;
        }
        const w = maxX - minX;
        const h = maxY - minY;
        if (w < 0.07 || h < 0.1) return false;
        const ar = w / Math.max(h, 1e-6);
        // 正脸略竖椭圆；侧脸可更窄。手掌误检常出现过宽或过扁
        if (ar < 0.38 || ar > 1.55) return false;
        return true;
    }

    captureFrame() {
        const face = this.sharedData.face;
        if (!face || face.length < 10) return null;
        if (!this._faceLandmarksBBoxLooksPlausible(face)) return null;

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
