import cv2
import numpy as np
from src.model import MLP

class_name = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def load_model():
    try:
        layer_dims = [784, 384, 384, 47]
        mlp = MLP(layer_dims=layer_dims, init='he', use_batchnorm=True)
        weights = np.load('models/mlp_weights_tuned.npz')
        loaded_params, loaded_bn_params = {}, {}
        for key in weights.keys():
            if key.startswith('gamma') or key.startswith('beta') or key.startswith('running_'):
                loaded_bn_params[key] = weights[key]
            elif key.startswith('W') or key.startswith('b'):
                loaded_params[key] = weights[key]
        mlp.parameters, mlp.bn_params = (loaded_params, loaded_bn_params)
        print("Model loaded successfully!")
        return mlp
    except Exception as e:
        print(f"Fail to load model: {str(e)}")
        return None

def preprocess_image(img_28x28):
    if np.sum(img_28x28) == 0:
        return None
    img_rotated = cv2.rotate(img_28x28, cv2.ROTATE_90_CLOCKWISE)
    
    img_flipped = cv2.flip(img_rotated, 1)

    img_normalized = img_flipped.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.13) / 0.30
    
    img_flattened = img_normalized.flatten()
    img_input = img_flattened.reshape(-1, 1)
    
    return img_input

def predict_character(model, processed_image):
    if processed_image is None:
        return None, None, None, None
        
    pred, _ = model.forward(processed_image, is_training=False)
    
    pred_class = np.argmax(pred, axis=0)[0]
    confidence = np.max(pred, axis=0)[0]
    pred_name = class_name[pred_class]
    
    top3_idxs = np.argsort(pred[:, 0])[-3:][::-1]
    top3_probs = pred[top3_idxs, 0]
    top3_preds = [class_name[i] for i in top3_idxs]
    
    return pred_name, confidence, top3_preds, top3_probs

CANVAS_SIZE = 28
ZOOM_FACTOR = 15
STROKE_WIDTH = 2
WINDOW_HEIGHT = CANVAS_SIZE * ZOOM_FACTOR
INFO_PANEL_WIDTH = 400
WINDOW_WIDTH = (CANVAS_SIZE * ZOOM_FACTOR) + INFO_PANEL_WIDTH

drawing = False
canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype="uint8")

def draw_on_canvas(event, x, y, flags, param):
    global drawing, canvas

    if x >= CANVAS_SIZE * ZOOM_FACTOR:
        drawing = False
        return

    x_canvas = x // ZOOM_FACTOR
    y_canvas = y // ZOOM_FACTOR
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        param['last_pos'] = (x_canvas, y_canvas)
        cv2.circle(canvas, (x_canvas, y_canvas), STROKE_WIDTH-1, 255, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ix, iy = param['last_pos']
            cv2.line(canvas, (ix, iy), (x_canvas, y_canvas), 255, STROKE_WIDTH)
            cv2.circle(canvas, (x_canvas, y_canvas), STROKE_WIDTH-1, 255, -1)
            param['last_pos'] = (x_canvas, y_canvas)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def main():
    model = load_model()
    if model is None:
        return
        
    global canvas
    
    window_name = "Real-time Character Recognition | 'c' to clear, 'q' to quit"
    cv2.namedWindow(window_name)
    
    mouse_param = {'last_pos': (0, 0)}
    cv2.setMouseCallback(window_name, draw_on_canvas, mouse_param)
    
    print("="*50)
    print("INSTRUCTIONS:")
    print(" - Draw on the left (black) panel.")
    print(" - Press 'c' to clear.")
    print(" - Press 'q' or ESC to quit.")
    print("="*50)

    while True:
        canvas_zoomed = cv2.resize(canvas, (WINDOW_HEIGHT, WINDOW_HEIGHT), 
                                   interpolation=cv2.INTER_NEAREST)
        canvas_display = cv2.cvtColor(canvas_zoomed, cv2.COLOR_GRAY2BGR)

        info_panel = np.full((WINDOW_HEIGHT, INFO_PANEL_WIDTH, 3), 50, dtype="uint8")
        
        processed_img = preprocess_image(canvas)
        
        if processed_img is not None:
            pred, conf, top3_p, top3_c = predict_character(model, processed_img)
            
            cv2.putText(info_panel, "Top Prediction:", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.putText(info_panel, f"{pred}", (150, 140), cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 255, 0), 5)
            cv2.putText(info_panel, f"Confidence: {conf:.1%}", (20, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

            cv2.putText(info_panel, "Top 3:", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            for i, (p, c) in enumerate(zip(top3_p, top3_c)):
                text = f"#{i+1}: {p} ({c:.0%})"
                cv2.putText(info_panel, text, (30, 315 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        else:
            cv2.putText(info_panel, "Draw a character...", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        
        combined_window = np.hstack([canvas_display, info_panel])
        
        cv2.imshow(window_name, combined_window)
        
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype="uint8")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()