from pynput import keyboard
import time

DEBUG = False
def pr(s):
    if DEBUG:
        print(s)


class KeyInput():
    def __init__(self):
        self.last_key = None
        self.consumed = False
        self.released = False

        # non-blocking listener
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        print("Keyboard listener: Started")

    def on_press(self, key):
        self.consumed = False
        self.released = False
        try:
            pr('key {0}'.format(key.char))
            self.last_key = key.char
        except AttributeError:
            pr('special key {0}'.format(key))
            self.last_key = key.name

    def on_release(self, key):
        self.released = True
        if self.consumed:
            self.last_key = None

        if key == keyboard.Key.esc:
            return False
        
    @property
    def key(self):
        return self.last_key if not self.released else None
    
    @property
    def latched(self):
        key = self.last_key

        self.consumed = True
        if self.released:
            self.last_key = None

        return key

    def close(self):
        self.listener.stop()
        self.listener.join()
        print("Keyboard listener: Closed")

class KeyManager:
    def __init__(self):
        self.ky = KeyInput()
        self.key = None

    def pool(self):
        self.key = self.ky.latched

    def close(self):
        self.ky.close()
    

if __name__ == '__main__':
    ky = KeyManager()
    sen = None
    while sen != 'q':
        ky.pool()
        sen = ky.key
        if sen is not None:
            print(sen)
        time.sleep(0.1)

    ky.close()