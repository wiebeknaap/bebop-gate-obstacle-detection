import cv2
import os
import numpy as np

image_root = "/home/pimovergaag/PycharmProjects/bebop-gate-obstacle-detection/data/raw"
images = sorted([img for img in os.listdir(image_root) if img.endswith(".jpg")])
for image in images:
    img = cv2.imread(os.path.join(image_root, image))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 100, 40])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    vertical_kernel = np.ones((9, 3), np.uint8)
    side_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(side_mask, connectivity=8)
    candidate_blobs = []
    debug_img = img.copy()

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        if area < 500:
            continue
        if h < 20:
            continue
        if h <= w:
            continue

        candidate_blobs.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'area': area,
            'cx': cx,
            'cy': cy,
        })
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        gate_pairs = []
        candidate_blobs = sorted(candidate_blobs, key=lambda blob: blob['cx'])
        for i in range(len(candidate_blobs)):
            for j in range(i + 1, len(candidate_blobs)):

                left_blob = candidate_blobs[i]
                right_blob = candidate_blobs[j]

                width_diff = abs(left_blob['w'] - right_blob['w'])
                if width_diff > 20:
                    continue

                height_diff = abs(left_blob['h'] - right_blob['h'])
                if height_diff > 50:
                    continue

                left_inner = left_blob['x'] + left_blob['w']
                right_inner = right_blob['x']
                opening_width = right_inner - left_inner
                if opening_width < 40:
                    continue

                if opening_width > 250:
                    continue

                y_diff = abs(left_blob['cy'] - right_blob['cy'])
                if y_diff > 40:
                    continue

                height_score = 1 - (height_diff / max(left_blob['h'], right_blob['h']))
                width_score = 1 - (width_diff / max(left_blob['w'], right_blob['w']))
                y_score = 1 - (y_diff / max(left_blob['h'], right_blob['h']))

                avg_height = 0.5 * (left_blob['h'] + right_blob['h'])
                ratio = opening_width / avg_height

                ideal_ratio = 1.0
                spacing_score = max(0, 1 - abs(ratio - ideal_ratio))

                score = (
                        0.4 * height_score +
                        0.2 * width_score +
                        0.2 * y_score +
                        0.2 * spacing_score
                )

                gate_pairs.append({
                    'left_blob': left_blob,
                    'right_blob': right_blob,
                    'opening_width': opening_width,
                    'score': score
                })



    cv2.imshow("mask", mask)
    cv2.imshow("blobs", debug_img)
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
