from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.clock import Clock
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

class VisionApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.camera = Camera(resolution=(640, 480), play=True)
        self.label = Label(text="Loading AI...", size_hint=(1, 0.2))
        layout.add_widget(self.camera)
        layout.add_widget(self.label)
        Clock.schedule_once(self.load_model, 1)
        return layout

    def load_model(self, dt):
        self.label.text = "Model loading (2 mins)..."
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.label.text = "Ready! Point camera at objects."
        Clock.schedule_interval(self.update, 5.0)

    def update(self, dt):
        texture = self.camera.texture
        if texture and hasattr(self, 'model'):
            img = Image.frombytes('RGBA', texture.size, texture.pixels).convert('RGB')
            inputs = self.processor(images=img, text="Describe this image", return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=30)
            self.label.text = self.processor.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    VisionApp().run()
