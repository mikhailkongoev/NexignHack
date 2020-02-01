import cv2
def drawrect(img, pt1, pt2, color, track_id, thickness=1, style='dotted'):
    if style == 'solid':
        cv2.rectangle(img, pt1, pt2, (0, 250, 0), thickness)
        cv2.putText(img, str(track_id), pt1, cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), thickness=4)
def draw_boxes_on_image_det(img, boxes):
    # left, top, right, bottom, track, phantom
    for box in boxes:
        left, top, right, bottom, confidence = box
        # color = (colours[(3 * track_id) % COLORS],
        #          colours[(3 * track_id + 1) % COLORS],
        #          colours[(3 * track_id + 2) % COLORS])
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))


        style = 'solid'

        color = 50
        track_id = 10

        drawrect(img, p1, p2, color, track_id, 3, style)
    return img

def draw_boxes_on_image_online(img, boxes):
    # left, top, right, bottom, track, phantom
    for box in boxes:
        left, top, right, bottom, track_id, phantom = box
        # color = (colours[(3 * track_id) % COLORS],
        #          colours[(3 * track_id + 1) % COLORS],
        #          colours[(3 * track_id + 2) % COLORS])
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))

        if phantom == 1:
            style = 'dashed'
            continue
        else:
            style = 'solid'

        color = 50

        drawrect(img, p1, p2, color, track_id, 3, style)
    return img