import cv2
import numpy as np
import argparse
from collections import deque

def clahe_gray(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return gray

def score_contour(cnt, frame_shape):
    # Pipe-ish heuristic: large area, elongated shape, not tiny noise.
    area = cv2.contourArea(cnt)
    if area < 3000:
        return -1e9

    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    (cx, cy), (w, h), ang = rect
    w = max(w, 1.0)
    h = max(h, 1.0)
    aspect = max(w, h) / min(w, h)

    # Penalize weird shapes:
    # - too "square"
    # - too close to tiny
    aspect_score = min(aspect, 20.0)  # cap
    area_score = np.log(area + 1.0)

    H, W = frame_shape[:2]
    # prefer contours not hugging image borders too much (optional)
    border_penalty = 0.0
    if cx < 10 or cy < 10 or cx > W - 10 or cy > H - 10:
        border_penalty = 0.5

    # Weighted score: tuned to favor long cylinder-like blobs
    return (2.0 * area_score) + (3.0 * aspect_score) - border_penalty

def rect_to_box(rect):
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    return box

def ema_rect(prev_rect, new_rect, alpha=0.2):
    """Exponential moving average on minAreaRect representation."""
    if prev_rect is None:
        return new_rect
    (pcx, pcy), (pw, ph), pa = prev_rect
    (ncx, ncy), (nw, nh), na = new_rect

    cx = (1 - alpha) * pcx + alpha * ncx
    cy = (1 - alpha) * pcy + alpha * ncy
    w  = (1 - alpha) * pw  + alpha * nw
    h  = (1 - alpha) * ph  + alpha * nh

    # angle wrap handling (keep it simple)
    # bring na close to pa by adding/subtracting 180
    while na - pa > 90: na -= 180
    while pa - na > 90: na += 180
    a  = (1 - alpha) * pa  + alpha * na

    return ((cx, cy), (w, h), a)

def make_pipe_mask(frame_shape, rect, thickness_extra=12):
    """Create a filled mask from minAreaRect, slightly thickened."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if rect is None:
        return mask

    (cx, cy), (w, h), ang = rect
    # Expand the minor axis a bit to cover edges/partial occlusion
    if w < h:
        w2, h2 = w + thickness_extra, h  # w is minor axis
    else:
        w2, h2 = w, h + thickness_extra  # h is minor axis

    rect2 = ((cx, cy), (w2, h2), ang)
    box = rect_to_box(rect2)
    cv2.fillPoly(mask, [box], 255)
    return mask

def overlay_mask(bgr, mask, alpha=0.35):
    """Overlay a green-ish highlight without hardcoding a specific palette style too much."""
    overlay = bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)  # BGR green
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input video path")
    ap.add_argument("--out", dest="out", default=None, help="Output video path")
    ap.add_argument("--show", action="store_true", help="Show playback window")
    ap.add_argument("--scale", type=float, default=1.0, help="Resize factor (e.g., 0.5 for faster)")
    ap.add_argument("--canny1", type=int, default=60, help="Canny threshold 1")
    ap.add_argument("--canny2", type=int, default=160, help="Canny threshold 2")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.inp}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.scale != 1.0:
        W2 = int(W * args.scale)
        H2 = int(H * args.scale)
    else:
        W2, H2 = W, H

    out_path = args.out
    if out_path is None:
        out_path = args.inp.rsplit(".", 1)[0] + "_pipe_highlight.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (W2, H2))

    prev_rect = None
    recent_rects = deque(maxlen=5)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.scale != 1.0:
            frame = cv2.resize(frame, (W2, H2), interpolation=cv2.INTER_AREA)

        gray = clahe_gray(frame)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray_blur, args.canny1, args.canny2)
        # Close small gaps so pipe edges become a coherent contour
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
        edges2 = cv2.dilate(edges2, None, iterations=1)

        contours, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        best_score = -1e18
        for cnt in contours:
            sc = score_contour(cnt, frame.shape)
            if sc > best_score:
                best_score = sc
                best_rect = cv2.minAreaRect(cnt)

        # If detection fails, fall back to a recent rect (helps when pipe vanishes briefly)
        if best_rect is not None and best_score > -1e8:
            recent_rects.append(best_rect)
        else:
            if len(recent_rects) > 0:
                best_rect = recent_rects[-1]

        # Smooth across frames to reduce jitter
        smoothed = ema_rect(prev_rect, best_rect, alpha=0.2) if best_rect is not None else prev_rect
        prev_rect = smoothed

        mask = make_pipe_mask(frame.shape, smoothed, thickness_extra=18)
        vis = overlay_mask(frame, mask, alpha=0.35)

        # Draw outline
        if smoothed is not None:
            box = rect_to_box(smoothed)
            cv2.polylines(vis, [box], True, (0, 255, 0), 2)

        writer.write(vis)

        if args.show:
            cv2.imshow("Pipe Highlight", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
