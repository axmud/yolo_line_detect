import cv2
import numpy as np
import carla
import hashlib

last_hash = None
count_rgb = 0

img_hash = 1
creating = True


# ---- Client is set inside local machine at port 2000 ----
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# --- Reset world and TM (to run again without closing CarlaUE4.exe) ---
town_name = 'Town10HD'
world = client.load_world(town_name)

static_weathers = [ carla.WeatherParameters.ClearNoon,
                    carla.WeatherParameters.CloudyNoon,
                    carla.WeatherParameters.WetNoon,
                    carla.WeatherParameters.WetCloudyNoon,
                    carla.WeatherParameters.MidRainyNoon,
                    carla.WeatherParameters.HardRainNoon,
                    carla.WeatherParameters.SoftRainNoon,
                    carla.WeatherParameters.ClearSunset,
                    carla.WeatherParameters.CloudySunset,
                    carla.WeatherParameters.WetSunset,
                    carla.WeatherParameters.WetCloudySunset,
                    carla.WeatherParameters.MidRainSunset,
                    carla.WeatherParameters.HardRainSunset,
                    carla.WeatherParameters.SoftRainSunset]



# try:
#     current_map = world.get_map().name.split('/')[-1]
#     world = client.load_world(current_map)
#     print(f"Reloaded map: {current_map}")
# except RuntimeError as e:
#     print(f"Map reload failed: {e}")


# ---- Traffic manager is reset ----
tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(False)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
print(world.get_map().name)
print(client.get_available_maps())
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
def world_recreate(new_map, worl, setting, actors):
    for a in actors:
        a.destroy()
    setting.synchronous_mode = False
    worl.apply_settings(settings)
    worl.load_world(new_map)
    settings.synchronous_mode = True
    worl.apply_settings(settings)

# --- Traffic Manager setup ---
tm_port = tm.get_port()
tm.set_synchronous_mode(True)
tm.set_global_distance_to_leading_vehicle(1.0)
tm.global_percentage_speed_difference(0)

# ---- setting up the instance segmentation camera ----
cmr_ins_seg = blueprint_library.find('sensor.camera.instance_segmentation')
cmr_ins_seg.set_attribute('role_name', 'camera')
cmr_ins_seg.set_attribute('image_size_x', '800')
cmr_ins_seg.set_attribute('image_size_y', '600')
cmr_ins_seg.set_attribute('fov', '90')

# ---- setting up the rgb camera ----
cmr_rgb = blueprint_library.find('sensor.camera.rgb')
cmr_rgb.set_attribute('role_name', 'camera')
cmr_rgb.set_attribute('image_size_x', '800')
cmr_rgb.set_attribute('image_size_y', '600')
cmr_rgb.set_attribute('fov', '90')

# Create semantic segmentation camera
bp_seg = blueprint_library.find('sensor.camera.semantic_segmentation')
bp_seg.set_attribute('image_size_x', '800')
bp_seg.set_attribute('image_size_y', '600')
bp_seg.set_attribute('fov', '90')





# ---- relative transform of the camera of ego vehicle ----
relative_transform = carla.Transform(
    carla.Location(x=1.5, z=1.7),
    carla.Rotation(pitch=0)
)

ego_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
ego_bp.set_attribute('role_name', 'hero')
def save_image(image, image_numbers, weather_number):
    global count_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    rgb = array[:, :, :3]
    cv2.imshow("CARLA Camera View", rgb)
    cv2.waitKey(1)
    count_rgb += 1
    if count_rgb == 20 and image_numbers < 100 and creating:
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        filename = f"dataset/rgb/{town_name}_{weather_number}_{image_numbers}.png"
        cv2.imwrite(filename, img)
        print("image", image_numbers)


def semantic_callback(image, image_numbers, weather_number):
    global count_rgb, town_name
    # Raw data is a flat uint8 array. Each pixel = 4 bytes (BGRA).
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))

    # The R channel contains the semantic tag ID directly.
    # (CARLA encodes class IDs into the red channel)
    semantic_map = img_array[:, :, 2]  # R channel in BGRA order
    lane_id = 24  # lane marking semantic ID (check below)
    lane_mask = (semantic_map == lane_id).astype(np.uint8) * 255

    # Now semantic_map[y,x] is the class ID (integer)
    # Example: 7 = lane marking, 1 = building, etc.
    # cv2.imwrite(f"images/lane_{image.frame}.png", lane_mask)
    if count_rgb == 20 and image_numbers < 100 and creating:
        filename = f"dataset/seg/{town_name}_{weather_number}_{image_numbers}.png"
        cv2.imwrite(filename, lane_mask)
        print("seg")
        count_rgb = 0
        return True
    else:
        return False
