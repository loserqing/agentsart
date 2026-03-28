/**
 * 负责摄像头捕捉、MediaPipe 初始化及图像截取
 */
export class VisionSystem {
    constructor(videoElement, sharedData) {
        this.video = videoElement;
        this.sharedData = sharedData;
        // 优化：预创建 Canvas 和 Context，避免每次捕捉时重复创建 DOM 节点
        this.captureCanvas = document.createElement('canvas');
        this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: true });
    }

    async init() {
        const holistic = new Holistic({ locateFile: (f) => `./assets/mediapipe/${f}` });
        holistic.setOptions({
            modelComplexity: 1,
            minDetectionConfidence: 0.55,
            minTrackingConfidence: 0.62,
            refineFaceLandmarks: true,
            smoothLandmarks: true,
        });
        
        holistic.onResults(this.handleResults.bind(this));
        
        const camera = new Camera(this.video, {
            onFrame: async () => { await holistic.send({ image: this.video }); },
            width: 640,
            height: 480
        });
        await camera.start();
    }

    handleResults(res) {
        const face = res.faceLandmarks;
        this.sharedData.face = face;
        this.sharedData.pose = res.poseLandmarks;

        if (this.video.videoWidth > 0) {
            this.sharedData.videoRes.x = this.video.videoWidth;
            this.sharedData.videoRes.y = this.video.videoHeight;
        }

        if (face) {
            this.sharedData.faceArea = Math.abs(face[10].y - face[152].y);
        } else {
            this.sharedData.faceArea = 0;
        }
    }
    
    captureFrame() {
        const maxRes = 512; // 优化：限制最大分辨率，加速AI处理
        
        // 如果有人脸数据，进行智能裁剪 (只捕捉头部)
        if (this.sharedData.face && this.sharedData.face.length > 0) {
            const face = this.sharedData.face;
            const vw = this.video.videoWidth;
            const vh = this.video.videoHeight;

            // 1. 计算人脸包围盒 (归一化坐标)
            let minX = 1, minY = 1, maxX = 0, maxY = 0;
            for (const pt of face) {
                if (pt.x < minX) minX = pt.x;
                if (pt.x > maxX) maxX = pt.x;
                if (pt.y < minY) minY = pt.y;
                if (pt.y > maxY) maxY = pt.y;
            }

            // 2. 计算像素中心与尺寸
            const cx = ((minX + maxX) / 2) * vw;
            const cy = ((minY + maxY) / 2) * vh;
            const w = (maxX - minX) * vw;
            const h = (maxY - minY) * vh;

            // 3. 确定裁剪框 (正方形，基于最大边长，扩大 2.2 倍以包含头发/颈部)
            const size = Math.max(w, h) * 2.2;

            // 缩放到 maxRes
            this.captureCanvas.width = Math.min(size, maxRes);
            this.captureCanvas.height = Math.min(size, maxRes);
            this.captureCtx.drawImage(this.video, cx - size / 2, cy - size / 2, size, size, 0, 0, this.captureCanvas.width, this.captureCanvas.height);
        } else {
            // 无人脸时也进行缩放
            const scale = Math.min(1, maxRes / Math.max(this.video.videoWidth, this.video.videoHeight));
            this.captureCanvas.width = this.video.videoWidth * scale; 
            this.captureCanvas.height = this.video.videoHeight * scale;
            this.captureCtx.drawImage(this.video, 0, 0, this.video.videoWidth, this.video.videoHeight, 0, 0, this.captureCanvas.width, this.captureCanvas.height);
        }
        // 优化：使用 JPEG 格式并降低质量到 0.8，大幅减小 Base64 体积
        return this.captureCanvas.toDataURL('image/jpeg', 0.8);
    }
}