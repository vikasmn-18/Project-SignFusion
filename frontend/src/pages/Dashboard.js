
import React, { useEffect, useState, useRef, useCallback } from "react";
import axios from "axios";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { ASL_SIGN_EMOJIS, ASL_SIGN_EMOJI_MAP } from "../data/aslSignEmojis";
import helloVideo from "../assets/hello.mp4";
import helpVideo from "../assets/help.mp4";
import noVideo from "../assets/no.mp4";
import yesVideo from "../assets/yes.mp4";
import thankYouVideo from "../assets/thankyou.mp4";
import sorryVideo from "../assets/Sorry.mp4";
import finishedVideo from "../assets/finished.mp4";
import pleaseVideo from "../assets/please.mp4";
 

const API = "http://127.0.0.1:5000";

const ASL_WORD_VIDEO_MAP = {
  hello: { label: "Hello", video: helloVideo },
  help: { label: "Help", video: helpVideo },
  no: { label: "No", video: noVideo },
  yes: { label: "Yes", video: yesVideo },
  thankyou: { label: "Thank You", video: thankYouVideo },
  sorry: { label: "Sorry", video: sorryVideo },
  finished: { label: "Finished", video: finishedVideo },
  please: { label: "Please", video: pleaseVideo },
};

const ASL_WORD_VIDEO_PATTERNS = [
  { tokens: ["thank", "you"], key: "thankyou" },
  { tokens: ["hello"], key: "hello" },
  { tokens: ["help"], key: "help" },
  { tokens: ["no"], key: "no" },
  { tokens: ["yes"], key: "yes" },
  { tokens: ["thankyou"], key: "thankyou" },
  { tokens: ["sorry"], key: "sorry" },
  { tokens: ["finished"], key: "finished" },
  { tokens: ["please"], key: "please" },
];

const buildASLSequenceFromText = (text) => {
  const words = text.match(/[A-Za-z]+/g) || [];
  const sequence = [];
  let wordIndex = 0;

  while (wordIndex < words.length) {
    let consumed = 1;
    let matchedVideo = null;

    for (const pattern of ASL_WORD_VIDEO_PATTERNS) {
      const candidate = words
        .slice(wordIndex, wordIndex + pattern.tokens.length)
        .map((word) => word.toLowerCase());

      if (candidate.length === pattern.tokens.length && candidate.every((word, index) => word === pattern.tokens[index])) {
        matchedVideo = ASL_WORD_VIDEO_MAP[pattern.key];
        consumed = pattern.tokens.length;
        break;
      }
    }

    if (sequence.length > 0) {
      sequence.push({ type: "space", letter: " " });
    }

    if (matchedVideo) {
      sequence.push({
        type: "word",
        word: words.slice(wordIndex, wordIndex + consumed).join(" ").toLowerCase(),
        label: matchedVideo.label,
        video: matchedVideo.video,
      });
      wordIndex += consumed;
      continue;
    }

    words[wordIndex].toUpperCase().split("").forEach((letter) => {
      sequence.push({
        type: "letter",
        letter,
        image: `/asl/${letter}.png`,
      });
    });
    wordIndex += 1;
  }

  return sequence;
};

/* ─── ASL Alphabet Gesture Classifier ───
   Matches the 26 hand-sign reference images (a_.png – z_.png).
   MediaPipe landmark indices: 0=wrist, 1-4=thumb, 5-8=index,
   9-12=middle, 13-16=ring, 17-20=pinky. y=0 is top of image.
*/
const classifyASLGestureFromChart = (lm) => {
  const d = (a, b) => {
    const dx = lm[a].x - lm[b].x;
    const dy = lm[a].y - lm[b].y;
    const dz = (lm[a].z || 0) - (lm[b].z || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };
  // Palm size for normalization (wrist to middle MCP)
  const palm = d(0, 9) || 0.001;

  // ── Finger extended: tip above PIP (y=0 is top of image) ──
  const indexUp  = lm[8].y < lm[6].y;
  const middleUp = lm[12].y < lm[10].y;
  const ringUp   = lm[16].y < lm[14].y;
  const pinkyUp  = lm[20].y < lm[18].y;

  // ── Finger tightly curled: tip below MCP ──
  const indexCurled  = lm[8].y > lm[5].y;
  const middleCurled = lm[12].y > lm[9].y;
  const ringCurled   = lm[16].y > lm[13].y;
  const pinkyCurled  = lm[20].y > lm[17].y;

  // ── Thumb analysis ──
  const thumbTipX = lm[4].x, thumbIPX = lm[3].x, wristX = lm[0].x;
  const thumbTipDist = Math.abs(thumbTipX - wristX);
  const thumbIPDist  = Math.abs(thumbIPX - wristX);
  const thumbOut     = thumbTipDist > thumbIPDist * 1.15;
  const thumbUp      = lm[4].y < lm[3].y && lm[4].y < lm[2].y;

  // ── Key distances (normalized) ──
  const thumbIndexDist  = d(4, 8)  / palm;
  const thumbMiddleDist = d(4, 12) / palm;
  const thumbRingDist   = d(4, 16) / palm;

  const indexMiddleDist = d(8, 12) / palm;
  const middleRingDist  = d(12, 16) / palm;
  const ringPinkyDist   = d(16, 20) / palm;

  // Thumb touching fingertips
  const thumbTouchIndex  = thumbIndexDist < 0.18;
  const thumbTouchMiddle = thumbMiddleDist < 0.18;

  // Index bent (hook): tip below DIP but above MCP
  const indexBent = lm[8].y > lm[7].y && lm[8].y < lm[5].y;

  // Hand pointing sideways (for G, H, P, Q)
  const indexSideways = Math.abs(lm[8].y - lm[5].y) < 0.08;
  const handPointsDown = lm[8].y > lm[5].y + palm * 0.5;

  // Index crossed over middle
  const normalSpacing = lm[5].x < lm[9].x;
  const tipSpacing = lm[8].x < lm[12].x;
  const indexCrossedMiddle = normalSpacing !== tipSpacing;

  const extCount = [indexUp, middleUp, ringUp, pinkyUp].filter(Boolean).length;

  // ═══ Classification (ordered from most distinctive to least) ═══

  // ─── Y: thumb + pinky extended, others curled ───
  if (pinkyUp && !indexUp && !middleUp && !ringUp && thumbOut) {
    return { letter: "Y", confidence: 0.93 };
  }

  // ─── I: only pinky extended ───
  if (pinkyUp && !indexUp && !middleUp && !ringUp && !thumbOut) {
    return { letter: "I", confidence: 0.91 };
  }

  // ─── L: index up + thumb out making L shape ───
  if (indexUp && !middleUp && !ringUp && !pinkyUp && thumbOut && thumbUp) {
    return { letter: "L", confidence: 0.93 };
  }

  // ─── W: index + middle + ring extended, pinky down ───
  if (indexUp && middleUp && ringUp && !pinkyUp && !thumbOut) {
    if (indexMiddleDist > 0.10 && middleRingDist > 0.10) {
      return { letter: "W", confidence: 0.91 };
    }
    return { letter: "W", confidence: 0.83 };
  }

  // ─── F: thumb + index form circle, other 3 fingers up ───
  if (thumbTouchIndex && middleUp && ringUp && pinkyUp) {
    return { letter: "F", confidence: 0.91 };
  }

  // ─── B: all four fingers extended & together, thumb tucked ───
  if (indexUp && middleUp && ringUp && pinkyUp && !thumbOut) {
    if (indexMiddleDist < 0.20 && middleRingDist < 0.20) {
      return { letter: "B", confidence: 0.91 };
    }
    return { letter: "B", confidence: 0.82 };
  }

  // ─── 5 / open hand: all 5 digits out (used here: all 4 fingers + thumb = open) ───
  if (indexUp && middleUp && ringUp && pinkyUp && thumbOut) {
    return { letter: "B", confidence: 0.75 }; // Open hand closest to B
  }

  // ─── V: index + middle spread apart ───
  if (indexUp && middleUp && !ringUp && !pinkyUp && indexMiddleDist > 0.18) {
    return { letter: "V", confidence: 0.92 };
  }

  // ─── R: index + middle crossed ───
  if (indexUp && middleUp && !ringUp && !pinkyUp && indexCrossedMiddle) {
    return { letter: "R", confidence: 0.85 };
  }

  // ─── U: index + middle together ───
  if (indexUp && middleUp && !ringUp && !pinkyUp && indexMiddleDist < 0.14) {
    return { letter: "U", confidence: 0.90 };
  }

  // ─── K: index + middle extended, thumb between them ───
  if (indexUp && middleUp && !ringUp && !pinkyUp && thumbTouchMiddle) {
    return { letter: "K", confidence: 0.86 };
  }

  // ─── H: index + middle pointing sideways ───
  if (indexUp && middleUp && !ringUp && !pinkyUp && indexSideways) {
    return { letter: "H", confidence: 0.83 };
  }

  // ─── V/U fallback for 2 extended fingers ───
  if (indexUp && middleUp && !ringUp && !pinkyUp) {
    return indexMiddleDist > 0.14
      ? { letter: "V", confidence: 0.78 }
      : { letter: "U", confidence: 0.78 };
  }

  // ─── D: index extended, thumb+middle form circle ───
  if (indexUp && !middleUp && !ringUp && !pinkyUp && thumbTouchMiddle) {
    return { letter: "D", confidence: 0.88 };
  }

  // ─── X: index hooked/bent ───
  if (indexBent && !middleUp && !ringUp && !pinkyUp) {
    return { letter: "X", confidence: 0.83 };
  }

  // ─── G: index + thumb pointing sideways ───
  if (!middleUp && !ringUp && !pinkyUp && indexSideways && thumbOut) {
    return { letter: "G", confidence: 0.82 };
  }

  // ─── Q: index + thumb pointing down ───
  if (handPointsDown && !middleUp && !ringUp && !pinkyUp && thumbOut) {
    return { letter: "Q", confidence: 0.78 };
  }

  // ─── P: like K but hand points down ───
  if (handPointsDown && !ringUp && !pinkyUp) {
    return { letter: "P", confidence: 0.75 };
  }

  // ─── D fallback: just index extended ───
  if (indexUp && !middleUp && !ringUp && !pinkyUp) {
    return { letter: "D", confidence: 0.80 };
  }

  // ─── O: all fingertips close together forming circle ───
  if (extCount === 0 && thumbTouchIndex && thumbMiddleDist < 0.25) {
    return { letter: "O", confidence: 0.85 };
  }

  // ─── C: curved hand (partially bent fingers, thumb apart) ───
  if (extCount === 0) {
    const indexPartial = lm[8].y < lm[5].y && lm[8].y > lm[6].y;
    const thumbApart = thumbIndexDist > 0.20 && thumbIndexDist < 0.55;
    if (indexPartial && thumbApart) {
      return { letter: "C", confidence: 0.78 };
    }
  }

  // ─── Fist variants: A, S, E, T, N, M ───
  if (extCount === 0) {
    // A: thumb sticks UP beside the fist (thumb pointing upward, not folded across palm)
    if (thumbUp && thumbOut && lm[4].y < lm[8].y) {
      return { letter: "A", confidence: 0.87 };
    }

    // S: tight fist, thumb folded OVER curled fingers (thumb sits in front at mid-palm level)
    if (!thumbUp && thumbOut && lm[4].y < lm[5].y) {
      return { letter: "S", confidence: 0.83 };
    }

    // E: fingers partially curled and resting on thumb — fingertips visible between DIP and MCP
    const eTips = lm[8].y > lm[7].y && lm[8].y < lm[5].y &&
                  lm[12].y > lm[11].y && lm[12].y < lm[9].y;
    if (eTips && !thumbOut) {
      return { letter: "E", confidence: 0.78 };
    }

    // T: thumb tip pokes between index and middle MCP knuckles
    const thumbBetweenIM = (lm[4].x > Math.min(lm[5].x, lm[9].x) &&
                            lm[4].x < Math.max(lm[5].x, lm[9].x));
    if (thumbBetweenIM && lm[4].y < lm[5].y) {
      return { letter: "T", confidence: 0.82 };
    }

    // N: thumb between middle and ring knuckles
    const thumbBetweenMR = (lm[4].x > Math.min(lm[9].x, lm[13].x) &&
                            lm[4].x < Math.max(lm[9].x, lm[13].x));
    if (thumbBetweenMR) {
      return { letter: "N", confidence: 0.80 };
    }

    // M: thumb between ring and pinky knuckles
    const thumbBetweenRP = (lm[4].x > Math.min(lm[13].x, lm[17].x) &&
                            lm[4].x < Math.max(lm[13].x, lm[17].x));
    if (thumbBetweenRP) {
      return { letter: "M", confidence: 0.80 };
    }

    // S: default tight fist
    return { letter: "S", confidence: 0.70 };
  }

  return { letter: "?", confidence: 0 };
};

const classifyASLGesture = classifyASLGestureFromChart;





/* ─── WAV encoder: converts raw PCM Float32 chunks to a WAV Blob ─── */
function encodeWAV(floatChunks, sampleRate) {
  let totalLen = 0;
  for (const c of floatChunks) totalLen += c.length;
  const samples = new Float32Array(totalLen);
  let off = 0;
  for (const c of floatChunks) { samples.set(c, off); off += c.length; }

  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  ws(8, "WAVE");
  ws(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  ws(36, "data");
  v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

function Dashboard() {
  // ─── Global state ───
  const [mode, setMode] = useState("asl-to-text");
  const [status, setStatus] = useState(null);

  // ─── ASL → Text/Speech state ───
  const [gestureText, setGestureText] = useState("");
  const [currentLetter, setCurrentLetter] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [isCapturing, setIsCapturing] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [handDetected, setHandDetected] = useState(false);
  const [fingerCount, setFingerCount] = useState(0);
  const [detectedASLLetter, setDetectedASLLetter] = useState("");
  const [detectionProgress, setDetectionProgress] = useState(0);
  const [lastConfirmed, setLastConfirmed] = useState("");
  const [cameraError, setCameraError] = useState("");

  // ─── Camera / detection refs ───
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animFrameRef = useRef(null);
  const streamRef = useRef(null);
  const stableRef = useRef({ letter: "", frames: 0 });
  const cooldownRef = useRef(false);
  const lastDetectTimeRef = useRef(0);
  const lastResultsRef = useRef(null);
  const isCapturingRef = useRef(false);

  // ─── Speech/Text → ASL state ───
  const [inputText, setInputText] = useState("");
  const [aslSequence, setAslSequence] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);
  const committedSpeechRef = useRef("");
  const [aslPlaying, setAslPlaying] = useState(false);
  const [aslIndex, setAslIndex] = useState(-1);
  const aslPlaybackTimerRef = useRef(null);

  // ─── Communication logs ───
  const [logs, setLogs] = useState([]);
  const [logSearch, setLogSearch] = useState("");
  const [logFilter, setLogFilter] = useState("");

  // ─── Sidebar / navigation ───
  const [activeTab, setActiveTab] = useState("asl-to-text");

  // MetaMask support removed.

  // ─── Sync state → refs ───
  useEffect(() => { isCapturingRef.current = isCapturing; }, [isCapturing]);

  /* ───────────────────────────────
     System status polling
  ─────────────────────────────── */
  useEffect(() => {
    const poll = setInterval(() => {
      axios.get(`${API}/status`).then((r) => setStatus(r.data)).catch(() => {});
    }, 3000);
    return () => clearInterval(poll);
  }, []);

  // MetaMask initialization removed (not used).

  useEffect(() => (
    () => {
      if (aslPlaybackTimerRef.current) {
        clearTimeout(aslPlaybackTimerRef.current);
      }
    }
  ), []);

  /* ───────────────────────────────
     Mode sync
  ─────────────────────────────── */
  const switchMode = useCallback(
    (m) => {
      setMode(m);
      setActiveTab(m);
      axios.post(`${API}/mode`, { mode: m }).catch(() => {});
      if (m !== "asl-to-text") stopCapture();
    },
    // eslint-disable-next-line
    []
  );

  /* ═══════════════════════════════
     MediaPipe Hand Detection Init
  ═══════════════════════════════ */
  const initHandDetection = async () => {
    if (handLandmarkerRef.current) return true;
    try {
      setModelLoading(true);
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
      );
      const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
      handLandmarkerRef.current = handLandmarker;
      setModelReady(true);
      setModelLoading(false);
      return true;
    } catch (err) {
      console.error("Hand detection init failed:", err);
      setModelLoading(false);
      setCameraError("Failed to load hand detection model. Check internet connection.");
      return false;
    }
  };

  /* ═══════════════════════════════
     1. ASL → Text / Speech
  ═══════════════════════════════ */
  const startCapture = async () => {
    setCameraError("");
    const ok = await initHandDetection();
    if (!ok) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsCapturing(true);
      isCapturingRef.current = true;
      requestAnimationFrame(detectFrame);
    } catch (err) {
      console.error("Camera error:", err);
      setCameraError("Camera access denied. Please allow camera permission and reload.");
    }
  };

  const stopCapture = () => {
    isCapturingRef.current = false;
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCapturing(false);
    setHandDetected(false);
    setFingerCount(0);
    setDetectedASLLetter("");
    setDetectionProgress(0);
    lastResultsRef.current = null;
    stableRef.current = { letter: "", frames: 0 };
  };

  /* ─── Detection frame loop ─── */
  const detectFrame = () => {
    if (!isCapturingRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    const ctx = canvas.getContext("2d");
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
    const W = canvas.width;
    const H = canvas.height;

    // Draw mirrored video
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    // ML detection (throttled ~12 fps)
    const now = performance.now();
    if (handLandmarkerRef.current && now - lastDetectTimeRef.current > 83) {
      lastDetectTimeRef.current = now;
      try {
        const results = handLandmarkerRef.current.detectForVideo(video, now);
        lastResultsRef.current = results;
        if (results.landmarks && results.landmarks.length > 0) {
          const primaryLm = results.landmarks[0];
          const fingerCt = countFingers(primaryLm);
          setFingerCount(fingerCt);
          const aslResult = classifyASLGestureFromChart(primaryLm);
          setDetectedASLLetter(aslResult.letter);
          processDetection(aslResult.letter, aslResult.confidence);
          setHandDetected(true);
        } else {
          setHandDetected(false);
          setDetectedASLLetter("");
          stableRef.current = { letter: "", frames: 0 };
          setDetectionProgress(0);
          setFingerCount(0);
        }
      } catch (_) { /* skip frame */ }
    }

    // Draw landmarks (mirrored)
    if (lastResultsRef.current?.landmarks?.length > 0) {
      for (const lm of lastResultsRef.current.landmarks) {
        drawHand(ctx, lm, W, H);
      }
    }

    animFrameRef.current = requestAnimationFrame(detectFrame);
  };

  /* ─── Draw hand landmarks ─── */
  const drawHand = (ctx, lm, W, H) => {
    const CONNS = [
      [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
      [13,17],[17,18],[18,19],[19,20],[0,17],
    ];
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    for (const [a, b] of CONNS) {
      ctx.beginPath();
      ctx.moveTo((1 - lm[a].x) * W, lm[a].y * H);
      ctx.lineTo((1 - lm[b].x) * W, lm[b].y * H);
      ctx.stroke();
    }
    const tips = [4, 8, 12, 16, 20];
    for (let i = 0; i < lm.length; i++) {
      ctx.beginPath();
      ctx.arc((1 - lm[i].x) * W, lm[i].y * H, tips.includes(i) ? 6 : 3, 0, Math.PI * 2);
      ctx.fillStyle = tips.includes(i) ? "#ef4444" : "#22c55e";
      ctx.fill();
    }
  };

  /* ─── Count raised fingers ─── */
  const countFingers = (lm) => {
    let count = 0;
    const wrist = lm[0];
    const tipDx = Math.abs(lm[4].x - wrist.x);
    const ipDx = Math.abs(lm[3].x - wrist.x);
    if (tipDx > ipDx * 1.15) count++;
    const tipIds = [8, 12, 16, 20];
    const pipIds = [6, 10, 14, 18];
    for (let i = 0; i < 4; i++) {
      if (lm[tipIds[i]].y < lm[pipIds[i]].y) count++;
    }
    return count;
  };

  const processDetection = (letter, conf) => {
    if (cooldownRef.current) return;

    // No valid detection
    if (!letter || letter === "?") {
      stableRef.current = { letter: "", frames: 0 };
      setDetectionProgress(0);
      return;
    }

    // Stabilize: require the same letter for N consecutive frames
    const s = stableRef.current;
    if (letter === s.letter) {
      s.frames++;
      const needed = 18;
      setDetectionProgress(Math.min(s.frames / needed, 1));
      if (s.frames >= needed) {
        setGestureText((prev) => {
          const t = prev + letter;
          axios.post(`${API}/detect`, { letter, text: t }).catch(() => {});
          return t;
        });
        setCurrentLetter(letter);
        setConfidence(conf);
        setLastConfirmed(letter);

        s.frames = 0;
        cooldownRef.current = true;
        setTimeout(() => { cooldownRef.current = false; setLastConfirmed(""); }, 1500);
      }
    } else {
      stableRef.current = { letter, frames: 1 };
      setDetectionProgress(0);
    }
  };

  const clearGesture = () => {
    axios.get(`${API}/clear`).then(() => {
      setGestureText("");
      setCurrentLetter("");
      setConfidence(0);
    });
  };

  const backspaceLetter = () => {
    setGestureText((prev) => prev.slice(0, -1));
  };

  const speakText = (text) => {
    if (!text) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "en-US";
    window.speechSynthesis.speak(utter);
    axios.post(`${API}/speak`, { text }).catch(() => {});
  };

  /* ═══════════════════════════════
     2. Speech / Text → ASL
  ═══════════════════════════════ */
  const [micError, setMicError] = useState("");
  const [micStatus, setMicStatus] = useState("");
  const listeningRef = useRef(false);
  const speechRecRef = useRef(null);

  const requestAudioPermission = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMicError("Microphone access is not supported by this browser.");
      return false;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      return true;
    } catch (err) {
      console.error("Microphone permission error:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setMicError("Microphone access denied. Please allow microphone permission in browser settings.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setMicError("No microphone found. Connect a microphone and retry.");
      } else {
        setMicError("Microphone error: " + (err.message || err.name));
      }
      return false;
    }
  };

  const startListening = async () => {
    setMicError("");
    setMicStatus("Starting speech recognition…");

    // Stop any previous recognition session first
    if (speechRecRef.current) {
      try { speechRecRef.current.stop(); } catch (e) { /* ignore */ }
      speechRecRef.current = null;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setMicError("Speech recognition is not supported in this browser. Please use Chrome or Edge.");
      return;
    }

    const allowed = await requestAudioPermission();
    if (!allowed) {
      setMicStatus("");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      listeningRef.current = true;
      setIsListening(true);
      setMicStatus("🎤 Listening… speak now");
    };

    recognition.onresult = (event) => {
      let finalTranscript = "";
      let interimTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }
      const appendText = (base, addition) => {
        if (!addition) return base;
        const trimmed = addition.trim();
        if (!trimmed) return base;
        if (!base) return trimmed;
        return base.endsWith(" ") ? base + trimmed : base + " " + trimmed;
      };

      if (finalTranscript) {
        committedSpeechRef.current = appendText(committedSpeechRef.current, finalTranscript);
        setInputText(committedSpeechRef.current);
        setMicStatus("✅ " + finalTranscript.trim());
      } else if (interimTranscript) {
        setInputText(appendText(committedSpeechRef.current, interimTranscript));
        setMicStatus("🎤 " + interimTranscript);
      }
    };

    recognition.onerror = (event) => {
      // "aborted" and "no-speech" are normal during continuous listening — don't show as errors
      if (event.error === "aborted") return;
      if (event.error === "no-speech") {
        setMicStatus("🎤 No speech detected – keep talking…");
        return;
      }
      console.error("Speech recognition error:", event.error);
      if (event.error === "audio-capture") {
        setMicError(
          "Microphone unavailable or blocked. Please allow microphone access in your browser site permissions, then retry."
        );
      } else if (event.error === "not-allowed" || event.error === "permission-denied") {
        setMicError(
          "Microphone access denied. Allow microphone permission in your browser or operating system settings."
        );
      } else if (event.error === "network") {
        setMicError("Network error – speech recognition requires an internet connection.");
      } else {
        setMicError("Speech recognition error: " + event.error);
      }
      listeningRef.current = false;
      setIsListening(false);
      setMicStatus("");
      if (speechRecRef.current) {
        try { speechRecRef.current.stop(); } catch (e) { /* ignore */ }
        speechRecRef.current = null;
      }
    };

    recognition.onend = () => {
      // Auto-restart if user hasn't stopped manually
      if (listeningRef.current) {
        try {
          setTimeout(() => {
            if (listeningRef.current && speechRecRef.current) {
              speechRecRef.current.start();
            }
          }, 300);
        } catch (e) { /* ignore */ }
      } else {
        setIsListening(false);
        setMicStatus("");
      }
    };

    speechRecRef.current = recognition;
    try {
      recognition.start();
    } catch (e) {
      setMicError("Failed to start speech recognition: " + e.message);
    }
  };

  const stopListening = () => {
    listeningRef.current = false;
    if (speechRecRef.current) {
      speechRecRef.current.stop();
      speechRecRef.current = null;
    }
    setIsListening(false);
    setMicStatus("");
  };

  const convertToASL = () => {
    if (!inputText.trim()) return;
    const sequence = buildASLSequenceFromText(inputText);
    if (!sequence.length) return;

    axios.get(`${API}/text-to-asl`, { params: { text: inputText } }).catch(() => {});
    setAslSequence(sequence);
    playASL(sequence, inputText);
  };

  const speakSequenceItem = (item) => {
    if (!item) return;
    const phrase = item.type === "space"
      ? "space"
      : item.type === "word"
        ? item.label
        : item.letter;
    const u = new SpeechSynthesisUtterance(phrase);
    u.lang = "en-US";
    u.rate = 0.9;
    u.volume = 1.0;
    window.speechSynthesis.speak(u);
  };

  const playASL = (seq, fullText) => {
    if (aslPlaybackTimerRef.current) {
      clearTimeout(aslPlaybackTimerRef.current);
      aslPlaybackTimerRef.current = null;
    }
    window.speechSynthesis.cancel();
    setAslPlaying(true);
    setAslIndex(0);
    speakSequenceItem(seq[0]);
    let i = 0;

    const advance = () => {
      i++;
      if (i >= seq.length) {
        setAslPlaying(false);
        setAslIndex(-1);
        aslPlaybackTimerRef.current = null;
        // Speak the full sentence after spelling
        if (fullText) {
          setTimeout(() => {
            const u = new SpeechSynthesisUtterance(fullText);
            u.lang = "en-US";
            u.volume = 1.0;
            window.speechSynthesis.speak(u);
            axios.post(`${API}/speak`, { text: fullText }).catch(() => {});
          }, 600);
        }
        return;
      }
      setAslIndex(i);
      speakSequenceItem(seq[i]);
      const delay = seq[i]?.type === "word" ? 2200 : 700;
      aslPlaybackTimerRef.current = setTimeout(advance, delay);
    };

    const firstDelay = seq[0]?.type === "word" ? 2200 : 700;
    aslPlaybackTimerRef.current = setTimeout(advance, firstDelay);
  };

  /* ═══════════════════════════════
     3. Communication Logs
  ═══════════════════════════════ */
  const fetchLogs = useCallback(() => {
    axios
      .get(`${API}/logs`, { params: { search: logSearch, direction: logFilter } })
      .then((r) => setLogs(r.data.logs))
      .catch(() => {});
  }, [logSearch, logFilter]);

  useEffect(() => {
    if (activeTab === "logs") fetchLogs();
  }, [activeTab, fetchLogs]);

  const clearLogs = () => {
    axios.post(`${API}/logs/clear`).then(() => setLogs([]));
  };

  // MetaMask connect/disconnect removed.

  /* ═══════════════════════════════
     Render helpers
  ═══════════════════════════════ */
  const hwDot = (ok) => (
    <span className={`hw-dot ${ok ? "online" : "offline"}`} />
  );

  const activeAslReference =
    ASL_SIGN_EMOJI_MAP[detectedASLLetter] ||
    ASL_SIGN_EMOJI_MAP[lastConfirmed] ||
    null;

  /* ═══════════════════════════════
     RENDER
  ═══════════════════════════════ */
  return (
    <div className="sf-app">
      {/* ─── SIDEBAR ─── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-icon">🤟</span>
          <h2>SignFusion</h2>
          <span className="brand-sub">KS5419</span>
        </div>

        <nav className="sidebar-nav">
          <button
            className={activeTab === "asl-to-text" ? "active" : ""}
            onClick={() => switchMode("asl-to-text")}
          >
            <span className="nav-icon">🎥</span> ASL → Text / Speech
          </button>
          <button
            className={activeTab === "speech-to-asl" ? "active" : ""}
            onClick={() => switchMode("speech-to-asl")}
          >
            <span className="nav-icon">🎙️</span> Speech / Text → ASL
          </button>
          <button
            className={activeTab === "logs" ? "active" : ""}
            onClick={() => setActiveTab("logs")}
          >
            <span className="nav-icon">📋</span> Communication Logs
          </button>
          <button
            className={activeTab === "status" ? "active" : ""}
            onClick={() => setActiveTab("status")}
          >
            <span className="nav-icon">⚙️</span> System Status
          </button>
          {/* MetaMask support removed — button intentionally omitted */}
        </nav>

        {/* mini status */}
        <div className="sidebar-footer">
          <div className="mini-status">
            {hwDot(status?.hardware?.camera)} Camera
          </div>
          <div className="mini-status">
            {hwDot(status?.hardware?.arduino)} Arduino
          </div>
          <div className="mini-status mode-badge">
            Mode: <strong>{mode === "asl-to-text" ? "ASL→Text" : "Speech→ASL"}</strong>
          </div>
        </div>
      </aside>

      {/* ─── MAIN CONTENT ─── */}
      <main className="main-content">
        {/* ═══ Header ═══ */}
        <header className="top-bar">
          <h1>
            {activeTab === "asl-to-text" && "ASL Gesture Recognition"}
            {activeTab === "speech-to-asl" && "Speech / Text to ASL"}
            {activeTab === "logs" && "Communication Logs"}
            {activeTab === "status" && "System Status & Hardware"}
          </h1>
          <div className="top-bar-right">
            <span className="live-badge">
              <span className="pulse" /> LIVE
            </span>
          </div>
        </header>

        {/* ═══════════════════════════════════
            TAB: ASL → Text / Speech
        ═══════════════════════════════════ */}
        {activeTab === "asl-to-text" && (
          <div className="tab-content">
            {/* Camera error */}
            {cameraError && (
              <div className="card error-card">
                <p>⚠️ {cameraError}</p>
              </div>
            )}

            {/* row 1 : camera + detection panel */}
            <div className="grid-2">
              {/* Camera feed (real) */}
              <div className="card camera-card">
                <div className="card-header">
                  <h3>📹 Camera Feed</h3>
                  <span className={`status-chip ${isCapturing ? "on" : "off"}`}>
                    {isCapturing ? (handDetected ? "Hand Detected" : "No Hand") : modelLoading ? "Loading Model…" : "Idle"}
                  </span>
                </div>
                <div className="camera-wrapper">
                  <video ref={videoRef} style={{ display: "none" }} playsInline muted />
                  <canvas ref={canvasRef} className="camera-canvas" />
                  {!isCapturing && !modelLoading && (
                    <div className="camera-overlay">
                      <p className="muted">Press Start to open camera &amp; detect hand gestures</p>
                    </div>
                  )}
                  {modelLoading && (
                    <div className="camera-overlay">
                      <div className="loader" />
                      <p>Loading hand detection model…</p>
                    </div>
                  )}
                </div>
                <div className="card-actions">
                  {!isCapturing ? (
                    <button className="btn btn-primary" onClick={startCapture} disabled={modelLoading}>
                      ▶ Start Capture
                    </button>
                  ) : (
                    <button className="btn btn-danger" onClick={stopCapture}>
                      ⏹ Stop
                    </button>
                  )}
                </div>
              </div>

              {/* Detection panel */}
              <div className="card detected-card">
                <div className="card-header">
                  <h3>🔤 Detected Gesture</h3>
                </div>

                {/* Finger count + letter */}
                <div className="big-letter-display">
                  {lastConfirmed ? (
                    <span className="big-letter confirmed">{lastConfirmed}</span>
                  ) : (
                    <span className="big-letter">{currentLetter || "–"}</span>
                  )}
                </div>

                {/* Finger count indicator */}
                <div className="finger-count-row">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <span key={n} className={`finger-dot ${fingerCount >= n ? "active" : ""}`}>
                      {n}
                    </span>
                  ))}
                </div>
                <p className="finger-label">
                  {handDetected
                    ? `ASL Sign Detected: ${detectedASLLetter || "..."} (${fingerCount} finger${fingerCount !== 1 ? "s" : ""})`
                    : "Show ASL hand sign to camera"}
                </p>

                {/* Detection progress */}
                <div className="confidence-bar-wrap">
                  <label>Hold steady to confirm</label>
                  <div className="confidence-bar">
                    <div
                      className={`confidence-fill ${detectionProgress >= 1 ? "done" : ""}`}
                      style={{ width: `${detectionProgress * 100}%` }}
                    />
                  </div>
                  <span className="confidence-val">
                    {detectionProgress >= 1 ? "✓ Confirmed!" : `${(detectionProgress * 100).toFixed(0)}%`}
                  </span>
                </div>

                {/* ASL reference guide */}
                <div className="letter-group-section">
                  <div className="group-mapping">
                    {ASL_SIGN_EMOJIS.map((item) => (
                      <span
                        key={`ref-${item.letter}`}
                        className={`map-item ${detectedASLLetter === item.letter || lastConfirmed === item.letter ? "active" : ""}`}
                        title={item.cue}
                      >
                        <img
                          className="map-image"
                          src={item.imagePath}
                          alt={`ASL ${item.letter}`}
                          onError={(e) => { e.currentTarget.style.display = "none"; }}
                        />
                        <strong className="map-letter">{item.letter}</strong>
                        {item.motion ? <span className="map-motion-note">Motion</span> : null}
                      </span>
                    ))}
                  </div>

                </div>
              </div>
            </div>

            {/* row 2 : formed text + speech */}
            <div className="card text-output-card">
              <div className="card-header">
                <h3>📝 Formed Text (Concatenation)</h3>
                <div className="card-actions inline">
                  <button className="btn btn-sm btn-secondary" onClick={() => setGestureText((p) => p + " ")} disabled={!gestureText || gestureText.endsWith(" ")}>
                    ␣ Space
                  </button>
                  <button className="btn btn-sm btn-secondary" onClick={backspaceLetter} disabled={!gestureText}>
                    ⌫ Backspace
                  </button>
                  <button className="btn btn-sm btn-secondary" onClick={clearGesture}>
                    🗑 Clear
                  </button>
                  <button
                    className="btn btn-sm btn-accent"
                    onClick={() => speakText(gestureText)}
                    disabled={!gestureText}
                  >
                    🔊 Speak
                  </button>
                </div>
              </div>
              <div className="text-output">
                {gestureText ? (
                  <h2 className="formed-text">{gestureText}<span className="cursor-blink">|</span></h2>
                ) : (
                  <p className="muted">Show finger gestures to camera. Hold steady ~1.5s to confirm each letter.</p>
                )}
              </div>
              <p className="hint">
                Show real ASL hand signs to the camera. Hold steady ~1.5s to confirm each letter. Use the Space button to add spaces between words.
              </p>
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════
            TAB: Speech / Text → ASL
        ═══════════════════════════════════ */}
        {activeTab === "speech-to-asl" && (
          <div className="tab-content">
            {/* Input area */}
            <div className="card">
              <div className="card-header">
                <h3>🎤 Input – Speech or Text</h3>
              </div>
              <div className="input-group">
                <textarea
                  className="text-input"
                  rows={3}
                  placeholder="Type a message or use the microphone…"
                  value={inputText}
                  onChange={(e) => {
                    committedSpeechRef.current = e.target.value;
                    setInputText(e.target.value);
                  }}
                />
              </div>
              <div className="card-actions">
                {!isListening ? (
                  <button className="btn btn-primary" onClick={startListening}>
                    🎙️ Start Mic
                  </button>
                ) : (
                  <button className="btn btn-danger" onClick={stopListening}>
                    ⏹ Stop Mic
                  </button>
                )}
                <button
                  className="btn btn-accent"
                  onClick={convertToASL}
                  disabled={!inputText.trim()}
                >
                  🤟 Convert to ASL
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    if (aslPlaybackTimerRef.current) {
                      clearTimeout(aslPlaybackTimerRef.current);
                      aslPlaybackTimerRef.current = null;
                    }
                    window.speechSynthesis.cancel();
                    setAslPlaying(false);
                    setAslIndex(-1);
                    setInputText("");
                    committedSpeechRef.current = "";
                    setAslSequence([]);
                  }}
                >
                  🗑 Clear
                </button>
              </div>
              {isListening && (
                <div className="listening-indicator">
                  <span className="pulse red" /> {micStatus || "Listening… speak now"}
                </div>
              )}
              {micError && (
                <div style={{ color: "#f87171", fontSize: ".85rem", marginTop: 8, padding: "8px 12px", background: "rgba(220,38,38,.1)", borderRadius: 8 }}>
                  ⚠️ {micError}
                  <div style={{ marginTop: 6, color: "#fbbf24" }}>
                    Tip: Click the lock icon near the address bar, enable microphone access for this site, then reload if needed.
                  </div>
                </div>
              )}
            </div>

            {/* ASL output display */}
            {aslSequence.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h3>🤟 ASL Gesture Sequence</h3>
                  <span className="badge">{aslSequence.filter((item) => item.type !== "space").length} signs</span>
                </div>
                <div className="asl-grid">
                  {aslSequence.map((item, i) => {
                    const ref = item.type === "letter" ? ASL_SIGN_EMOJI_MAP[item.letter] : null;
                    return item.type === "space" ? (
                      <div key={i} className="asl-space">⎵</div>
                    ) : (
                      <div
                        key={i}
                        className={`asl-card ${item.type === "word" ? "word-card" : ""} ${aslPlaying && aslIndex === i ? "highlight" : ""}`}
                      >
                        {item.type === "word" ? (
                          <video
                            className="asl-video"
                            src={item.video}
                            muted
                            loop
                            autoPlay
                            playsInline
                            controls
                            preload="metadata"
                          />
                        ) : (
                          <img
                            className="asl-image"
                            src={item.image || ref?.imagePath}
                            alt={`ASL ${item.letter}`}
                            onError={(e) => { e.currentTarget.style.display = "none"; }}
                          />
                        )}
                        <div className="asl-symbol" title={item.type === "word" ? `${item.label} sign video` : (ref?.cue || `ASL ${item.letter}`)}>
                          {item.type === "word" ? "Word Sign" : (ref?.symbol || item.letter)}
                        </div>
                        <div className="asl-letter-box">{item.type === "word" ? item.label : item.letter}</div>
                        <small>{item.type === "word" ? `ASL video for ${item.label}` : (ref?.cue || `Sign ${i + 1}`)}</small>
                      </div>
                    )
                  })}
                </div>
                {aslPlaying && (
                  <div className="playing-indicator">
                    ▶ Playing ASL sequence…
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ═══════════════════════════════════
            TAB: Communication Logs
        ═══════════════════════════════════ */}
        {activeTab === "logs" && (
          <div className="tab-content">
            <div className="card">
              <div className="card-header">
                <h3>📋 Communication History</h3>
                <div className="card-actions inline">
                  <button className="btn btn-sm btn-secondary" onClick={clearLogs}>
                    🗑 Clear All
                  </button>
                  <button className="btn btn-sm btn-primary" onClick={fetchLogs}>
                    🔄 Refresh
                  </button>
                </div>
              </div>
              {/* Search & Filter */}
              <div className="log-filters">
                <input
                  className="search-input"
                  placeholder="Search logs…"
                  value={logSearch}
                  onChange={(e) => setLogSearch(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && fetchLogs()}
                />
                <select
                  className="filter-select"
                  value={logFilter}
                  onChange={(e) => { setLogFilter(e.target.value); }}
                >
                  <option value="">All Directions</option>
                  <option value="ASL → Text">ASL → Text</option>
                  <option value="Text → Speech">Text → Speech</option>
                  <option value="Speech → Text">Speech → Text</option>
                  <option value="Text → ASL">Text → ASL</option>
                </select>
              </div>
              {/* Table */}
              {logs.length === 0 ? (
                <p className="muted center">No communication logs yet.</p>
              ) : (
                <div className="log-table-wrap">
                  <table className="log-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Direction</th>
                        <th>Input</th>
                        <th>Output</th>
                      </tr>
                    </thead>
                    <tbody>
                      {logs.map((l) => (
                        <tr key={l.id}>
                          <td className="mono">{new Date(l.timestamp).toLocaleTimeString()}</td>
                          <td>
                            <span className="dir-badge">{l.direction}</span>
                          </td>
                          <td>{l.input}</td>
                          <td>{l.output}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════
            TAB: System Status
        ═══════════════════════════════════ */}
        {activeTab === "status" && (
          <div className="tab-content">
            <div className="grid-2">
              {/* Hardware status */}
              <div className="card">
                <div className="card-header">
                  <h3>🔌 Hardware Status</h3>
                </div>
                <div className="hw-list">
                  {status?.hardware &&
                    Object.entries(status.hardware).map(([k, v]) => (
                      <div className="hw-item" key={k}>
                        {hwDot(v)}
                        <span className="hw-name">{k.charAt(0).toUpperCase() + k.slice(1)}</span>
                        <span className={`hw-label ${v ? "online" : "offline"}`}>
                          {v ? "Online" : "Offline"}
                        </span>
                      </div>
                    ))}
                </div>
              </div>

              {/* System info */}
              <div className="card">
                <div className="card-header">
                  <h3>ℹ️ System Information</h3>
                </div>
                <div className="info-grid">
                  <div className="info-item">
                    <span className="info-label">Current Mode</span>
                    <span className="info-value">{mode === "asl-to-text" ? "ASL → Text/Speech" : "Speech/Text → ASL"}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Last Detected</span>
                    <span className="info-value">{status?.current_letter || "—"}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Confidence</span>
                    <span className="info-value">{status?.confidence ? `${(status.confidence * 100).toFixed(0)}%` : "—"}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Total Logs</span>
                    <span className="info-value">{status?.log_count ?? 0}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Buffer</span>
                    <span className="info-value">{status?.current_text || "Empty"}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Platform</span>
                    <span className="info-value">Arduino Uno + AI + Web</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Architecture overview */}
            <div className="card">
              <div className="card-header">
                <h3>🏗️ System Architecture</h3>
              </div>
              <div className="arch-flow">
                <div className="arch-box hw">
                  <strong>Hardware Layer</strong>
                  <small>Arduino Uno · LEDs · Buzzer · Buttons · Display</small>
                </div>
                <div className="arch-arrow">⟶</div>
                <div className="arch-box ai">
                  <strong>AI / Processing Layer</strong>
                  <small>OpenCV · MediaPipe · ML Model · TTS · STT</small>
                </div>
                <div className="arch-arrow">⟶</div>
                <div className="arch-box web">
                  <strong>Web Dashboard</strong>
                  <small>Live Monitoring · Logs · Data Storage · Controls</small>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default Dashboard;
