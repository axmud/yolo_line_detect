import queue
import random
import math
import cv2

MIN_SPEED_KMH = 1.0
image_numbers = 0


# ---- getting all required stuff ----
from dataset_creation_substuff import (world, tm, settings, blueprint_library, tm_port, spawn_points,
                           relative_transform, ego_bp, cmr_rgb, save_image, bp_seg,
                           semantic_callback, static_weathers)

# ---- Spawn traffic vehicles ----
actors = []
vehicle_blueprints = blueprint_library.filter('*vehicle*')
for i in range(10):
    v = world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
    if v:
        actors.append(v)

# ---- Spawn ego vehicle ----
while True:
    ego_vehicle = world.try_spawn_actor(ego_bp, random.choice(spawn_points))
    if ego_vehicle:
        actors.append(ego_vehicle)
        break

# ---- setting up autopilot
for v in actors:
    v.set_autopilot(True, tm_port)


rgb_queue = queue.Queue()
segm_queue = queue.Queue()

# ---- attaching camera to ego vehicle ----
# camera_ins = world.spawn_actor(cmr_ins_seg, relative_transform, attach_to=ego_vehicle)
camera_rgb = world.spawn_actor(cmr_rgb, relative_transform, attach_to=ego_vehicle)
# print(f"Camera {camera_ins.id} attached to vehicle {ego_vehicle.id}")

# ---- creating new window to show images from camera
image_numbers = 0
# camera_ins.listen(lambda image: save_image(image, typ="ins"))
camera_rgb.listen(rgb_queue.put)

cam_seg = world.spawn_actor(bp_seg, relative_transform, attach_to=ego_vehicle)
cam_seg.listen(segm_queue.put)


# ---- simulation will start ----
print("Simulation running... Press Ctrl+C to stop.")
a = 0
try:
    for i in static_weathers:
        world.set_weather(i)
        while True:
            world.tick()
            rgb_image = rgb_queue.get()
            seg_image = segm_queue.get()

            if rgb_image.frame != seg_image.frame:
                print(f"Frame mismatch: RGB {rgb_image.frame}, SEG {seg_image.frame}")
                continue

                # ---- compute vehicle speed ----
            vel = ego_vehicle.get_velocity()
            speed_ms = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            speed_kmh = speed_ms * 3.6

            # ---- skip saving if vehicle is basically stopped ----
            if speed_kmh < MIN_SPEED_KMH:
                # optional: print occasionally for debugging
                # print(f"Skipping frame {rgb_image.frame}, speed={speed_kmh:.2f} km/h")
                continue
            # ---- car is moving → save images ----
            save_image(rgb_image, image_numbers, a)  # your existing RGB save
            if semantic_callback(seg_image, image_numbers, a):
                image_numbers += 1
            if image_numbers == 100:
                image_numbers = 0
                break
        a+=1
except KeyboardInterrupt:
    print("Stopping simulation...")
finally:
    camera_rgb.stop()
    camera_rgb.destroy()
    # camera_ins.stop()
    # camera_ins.destroy()
    cam_seg.stop()
    cam_seg.destroy()
    for a in actors:
        a.destroy()
    tm.set_synchronous_mode(False)
    settings.synchronous_mode = False
    world.apply_settings(settings)
    cv2.destroyAllWindows()
    print("Clean shutdown complete.")