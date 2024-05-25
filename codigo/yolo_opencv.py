import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np
import time


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap1 = cv2.VideoCapture(2)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
  
    cap2 = cv2.VideoCapture(4)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
  



    model = YOLO("/home/walther/work/projetos/TCC/codigo/Calib-cam/best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while True:
        inicio = time.time()

        ret, fram1 = cap1.read()
        ret2, fram2 = cap2.read()


        # Calcula as novas dimensões
        new_width = int(fram1.shape[1] * scale_percent)
        new_height = int(fram1.shape[0] * scale_percent)

        # Redimensiona as imagens
        fram1 = cv2.resize(fram1, (new_width, new_height))
        fram2 = cv2.resize(fram2, (new_width, new_height))

        result1 = model(fram1, agnostic_nms=True)[0]
        result2 = model(fram2, agnostic_nms=True)[0]

        detections1 = sv.Detections.from_yolov8(result1)
        detections2 = sv.Detections.from_yolov8(result2)

        labels1 = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections1
        ]

        labels2 = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections2
        ]
        fram1 = box_annotator.annotate(
            scene=fram1, 
            detections=detections1, 
            labels=labels1
        )
        fram2 = box_annotator.annotate(
            scene=fram2, 
            detections=detections2, 
            labels=labels2
        )

        zone.trigger(detections=detections1)
        fram1 = zone_annotator.annotate(scene=fram1)   
        zone.trigger(detections=detections2)
        fram2 = zone_annotator.annotate(scene=fram2)      

        
        cv2.imshow("esquerda", fram1)
        cv2.imshow("direita", fram2)

        print("A função levou {:.5f} segundos para ser executada.".format(time.time()-inicio))
         # Verificar se o usuário pressionou a tecla 'q' para sair
        break
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            


if __name__ == "__main__":
    main()