from ultralytics import solutions
inf = solutions.Inference(
    model="models/yolo11m-finetune.pt/"  # use your fine-tuned model
)
inf.inference()