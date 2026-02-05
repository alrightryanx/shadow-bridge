import tkinter as tk
import random
import winsound
import ctypes

class ConfettiEffect:
    def __init__(self, root):
        self.root = root
        self.top = None
        self.canvas = None
        self.particles = []
        self.active = False

    def start(self, duration_ms=3000):
        if self.active:
            return
        
        self.active = True
        
        try:
            # Create a top-level transparent window for confetti
            self.top = tk.Toplevel(self.root)
            self.top.attributes("-topmost", True)
            self.top.attributes("-transparentcolor", "white")
            self.top.attributes("-fullscreen", True)
            self.top.config(bg="white")
            self.top.overrideredirect(True)
            
            # Make it non-interactive
            if hasattr(ctypes.windll.user32, "SetWindowLongW"):
                GWL_EXSTYLE = -20
                WS_EX_LAYERED = 0x80000
                WS_EX_TRANSPARENT = 0x20
                try:
                    hwnd = int(self.top.frame(), 16)
                    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
                except:
                    pass

            self.canvas = tk.Canvas(self.top, bg="white", highlightthickness=0, bd=0)
            self.canvas.pack(fill=tk.BOTH, expand=True)
            
            screen_width = self.top.winfo_screenwidth()
            screen_height = self.top.winfo_screenheight()
            
            colors = ["#f44336", "#e91e63", "#9c27b0", "#673ab7", "#3f51b5", "#2196f3", "#03a9f4", "#00bcd4", "#009688", "#4caf50", "#8bc34a", "#cddc39", "#ffeb3b", "#ffc107", "#ff9800", "#ff5722"]
            
            for _ in range(100):
                x = random.randint(0, screen_width)
                y = random.randint(-screen_height, 0)
                size = random.randint(5, 15)
                color = random.choice(colors)
                speed_y = random.uniform(2, 7)
                speed_x = random.uniform(-2, 2)
                
                p = self.canvas.create_rectangle(x, y, x + size, y + size, fill=color, outline="")
                self.particles.append({
                    "id": p,
                    "x": x,
                    "y": y,
                    "size": size,
                    "speed_x": speed_x,
                    "speed_y": speed_y
                })
                
            self._animate()
            self.root.after(duration_ms, self.stop)
        except Exception as e:
            self.active = False

    def _animate(self):
        if not self.active or not self.top:
            return
            
        try:
            screen_height = self.top.winfo_screenheight()
            screen_width = self.top.winfo_screenwidth()
            
            for p in self.particles:
                p["y"] += p["speed_y"]
                p["x"] += p["speed_x"]
                
                if p["y"] > screen_height:
                    p["y"] = -p["size"]
                    p["x"] = random.randint(0, screen_width)
                
                self.canvas.coords(p["id"], p["x"], p["y"], p["x"] + p["size"], p["y"] + p["size"])
                
            self.root.after(20, self._animate)
        except:
            self.stop()

    def stop(self):
        self.active = False
        try:
            if self.top:
                self.top.destroy()
                self.top = None
        except:
            pass
        self.particles = []

def play_ping_sound():
    try:
        # Use a nice system sound
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception:
        # Fallback to simple beep
        try:
            ctypes.windll.user32.MessageBeep(0)
        except Exception:
            pass
