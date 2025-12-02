import carla
import random
import time
import numpy as np
import cv2
from ultralytics import YOLO


weather_options = [
        carla.WeatherParameters.ClearNoon,
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
tm_port = 8000
sprawn_vehicles = []
sprawn_vehicles_num = 50
IMG_HEIGHT = 600
IMG_WIDTH =800
relative_transform = carla.Transform(carla.Location(x=1.5, z=1.7),carla.Rotation(pitch=0))
latest_image = {"data": None}
weather_index = 0
maps_list = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"]
map_index = 0


def yolo_detect_lines(model, frame_bgr):
    """
    Runs YOLO on frame and returns frame with boxes drawn.
    """
    results = model.predict(frame_bgr, imgsz=640, conf=0.30, verbose=False)

    detections = results[0].boxes

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        # Draw box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        label = f"{cls_id}: {conf:.2f}"
        cv2.putText(frame_bgr, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame_bgr


def load_new_world(client, town_name):
    new_world = client.load_world(town_name)
    settings = new_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    new_world.apply_settings(settings)
    return new_world


def destroy_vehicles():
    global sprawn_vehicles
    try:
        for vehicle in list(sprawn_vehicles):  # iterate over a copy
            try:
                vehicle.destroy()
            except Exception as e:
                print("Error destroying vehicle:", e)
        sprawn_vehicles.clear()
    except Exception as e:
        print("error in destroy_vehicles: ", e)
    print("Vehicles are destroyed")


def destroy_camera(camera):
    try:
        # stop stream & detach callback
        camera.stop()
        camera.listen(lambda image: None)  # clear callback
        time.sleep(0.1)  # small delay to let C++ side flush
    except Exception as e:
        print("Error stopping camera:", e)
    try:
        camera.destroy()
    except Exception as e:
        print("Error destroying camera:", e)
    print("Camera is destroyed")


def change_map(camera_rgb):
    """
    Destroy current actors, load next map, re-create everything.
    Returns: new_world, new_camera_rgb, new_ego_vehicle
    """
    global maps_list, map_index

    destroy_camera(camera_rgb)
    destroy_vehicles()

    new_client = connect_carla_and_create_client()

    print(f"Loading new world: {maps_list[map_index]}")
    new_world = load_new_world(new_client, maps_list[map_index])

    set_weather(new_world)

    new_tm = create_traffic_manager(new_client)

    create_sprawn_vehicles(new_world)

    new_ego_vehicle = create_ego_vehicle(new_world)

    new_camera_rgb = create_rgb_camera(new_world, new_ego_vehicle)

    listen_rgb_camera(new_camera_rgb)

    print(f"Map: {maps_list[map_index]}")

    map_index = map_index + 1
    if map_index == len(maps_list):
        map_index = 0

    return new_world, new_camera_rgb, new_ego_vehicle


def change_weather(world):
    global weather_index, weather_options
    world.set_weather(weather_options[weather_index])
    print("Weather changed to {}".format(weather_index))
    weather_index += 1
    if weather_index >= len(weather_options):
        weather_index = 0


def show_camera_image(image):
    global latest_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    bgr = array[:, :, :3].copy()
    latest_image["data"] = bgr


def listen_rgb_camera(camera_rgb):
    camera_rgb.listen(lambda image: show_camera_image(image))


def warmup_world(world):
    for _ in range(5):
        world.tick()
        time.sleep(0.01)


def create_rgb_camera(world, vehicle):
    global IMG_HEIGHT, IMG_WIDTH
    blueprint_library = world.get_blueprint_library()
    cmr_rgb = blueprint_library.find('sensor.camera.rgb')
    cmr_rgb.set_attribute('role_name', 'camera')
    cmr_rgb.set_attribute('image_size_x', str(IMG_WIDTH))
    cmr_rgb.set_attribute('image_size_y', str(IMG_HEIGHT))
    cmr_rgb.set_attribute('fov', '90')
    camera_rgb = world.spawn_actor(cmr_rgb, relative_transform, attach_to=vehicle)
    return camera_rgb


def create_ego_vehicle(world, autopilot=True):
    global sprawn_vehicles, tm_port
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    while True:
        try:
            vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
            if vehicle:
                sprawn_vehicles.append(vehicle)
                print("Ego vehicle created")
                break
        except Exception as e:
            print(f"There is an error while creating ego vehicle :{e}\nTrying again...", flush=True)
    if autopilot:
        vehicle.set_autopilot(True, tm_port)
    return vehicle


def create_sprawn_vehicles(world):
    global sprawn_vehicles, tm_port
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for _ in range(sprawn_vehicles_num):
        try:
            vehicle = world.spawn_actor(random.choice(vehicle_bp), random.choice(spawn_points))
            if vehicle:
                sprawn_vehicles.append(vehicle)
        except RuntimeError:
            continue
    for vehicle in sprawn_vehicles:
        vehicle.set_autopilot(True, tm_port)
    print(f"{len(sprawn_vehicles)} vehicles spawned.")


def create_traffic_manager(client):
    global tm_port
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.0)
    tm.global_percentage_speed_difference(0)
    return tm


def set_weather(world):
    """
    Randomly choose one of your preferred weather presets.
    """
    global weather_options
    weather = random.randint(0, len(weather_options) - 1)
    world.set_weather(weather_options[weather])
    print("Weather set to {}".format(weather))


def create_world(client):
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS sim
    world.apply_settings(settings)
    return world


def connect_carla_and_create_client():
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    return client


def main():
    client = connect_carla_and_create_client()

    world = create_world(client)

    set_weather(world)

    tm = create_traffic_manager(client)

    create_sprawn_vehicles(world)

    ego_vehicle = create_ego_vehicle(world)

    camera_rgb = create_rgb_camera(world, ego_vehicle)

    yolo_model = YOLO("best.pt")

    try:
        warmup_world(world)

        listen_rgb_camera(camera_rgb)

        print("Starting visualization loop.")
        print("q / ESC  : quit")
        print("e        : change weather")
        print("m        : change map")

        while True:
            world.tick()

            frame = latest_image["data"]

            if frame is not None:
                frame = yolo_detect_lines(yolo_model, frame)
                cv2.imshow("Lane detection (autopilot, q/ESC to quit, w weather, m map)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: break
            elif key == ord('e'): change_weather(world)
            elif key == ord('m'):
                world, camera_rgb, ego_vehicle = change_map(camera_rgb)



    except Exception as e:
        print("Error in Main try-except block: ",e)
    finally:
        destroy_camera(camera_rgb)
        destroy_vehicles()
        cv2.destroyAllWindows()

        print("Clean shutdown complete.")


if __name__ == "__main__":
    main()
