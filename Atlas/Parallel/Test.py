import pycuda.driver as drv
drv.init()
dev = drv.Device(0)
print("Async copy engines:", dev.get_attribute(drv.device_attribute.ASYNC_ENGINE_COUNT))
