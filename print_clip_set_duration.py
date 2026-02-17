import inspect
from moviepy.Clip import Clip
print('has set_duration', hasattr(Clip, 'set_duration'))
print('method', Clip.set_duration if hasattr(Clip,'set_duration') else None)
print('has with_duration', hasattr(Clip, 'with_duration'))
