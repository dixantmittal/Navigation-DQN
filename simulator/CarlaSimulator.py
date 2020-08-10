import random
import time

import numpy

import carla
from constants import *
from simulator.ISimulator import ISimulator


# Random spawn location.
# These numbers are valid for only a particular junction scenario.
def random_location(rand):
    # Spawn on left side
    if rand < 0.5:
        return carla.Transform(carla.Location(x=random.randint(20, 60), y=random.choice([130.5, 127.5]), z=0.2), carla.Rotation(yaw=180))
    # Spawn on right side
    else:
        return carla.Transform(carla.Location(x=-random.randint(20, 60), y=random.choice([134.5, 137.5]), z=0.2), carla.Rotation(yaw=0))


class CarlaSimulator(ISimulator):
    def __init__(self, args):
        super().__init__(args)

        self.collided = True
        self.nVehicles = args.nVehicles

        self.av = None
        self.gridSensor = None
        self.collisionSensor = None
        self.vehicles = []
        self.frames = []

        # initialise Carla client
        self.world = carla.Client(args.host, args.port).get_world()
        self.bplib = self.world.get_blueprint_library()

        #         self.waypoints = numpy.rint(numpy.unique(numpy.rint([[point.transform.location.x, point.transform.location.y, point.transform.rotation.yaw] for point in self.world.get_map().generate_waypoints(0.5) if -50 < point.transform.location.x < 50 and 90 < point.transform.location.y < 190]), axis=0))
        self.destination = random.choice([[-6, 140, 90], [-50, 130, 180], [50, 137, 0]])

    def reset(self):
        self.destroy()

        # Spawn Autonomous car.
        # Spawn position is random in a certain range. Numbers valid for only particular junction.
        self.av = self.world.spawn_actor(self.bplib.find('vehicle.tesla.model3'), carla.Transform(carla.Location(-6, random.randint(100, 120), 0), carla.Rotation(yaw=90)))
        # Apply brake as default control
        self.av.apply_control(carla.VehicleControl(brake=1))

        # Choose a random number of NPC to spawn
        vehicles_to_spawn = random.randint(1, self.nVehicles)
        tries = 0
        while vehicles_to_spawn > 0 or tries < vehicles_to_spawn * 10:

            # Spawn NPC
            vehicle = self.world.try_spawn_actor(self.bplib.find('vehicle.audi.a2'), random_location(numpy.random.rand()))

            # if spawning failed, retry
            if vehicle is None:
                continue

            self.vehicles.append(vehicle)
            vehicles_to_spawn -= 1
            tries += 1

        # Spawn Collision detector.
        self.collisionSensor = self.world.spawn_actor(self.bplib.find('sensor.other.collision'), carla.Transform(), attach_to=self.av)
        self.collisionSensor.listen(lambda x: self.onCollision())
        self.collided = False

        # Wait for the simulator to get ready
        time.sleep(1 / FREQUENCY)

        self.frames = [self.locations()] * nFRAMES

        return self.state()

    def step(self, a):
        # apply default control to NPCs
        for vehicle in self.vehicles:
            vehicle.apply_control(carla.VehicleControl(throttle=0.5))

        # apply action to AV
        self.av.apply_control(carla.VehicleControl(throttle=THROTTLE[a], brake=BRAKE[a], manual_gear_shift=True, gear=1))

        # sleep to observe action's effect
        time.sleep(1 / FREQUENCY)

        # calculate reward
        reward = self.av.get_velocity().y * (-100 if self.collided else 1) - 1

        self.frames.append(self.locations())

        return self.state(), reward, self.collided, None

    def onCollision(self):
        self.collided = True

    def state(self):
        self.frames = self.frames[-nFRAMES:]
        return numpy.array(self.frames).astype(numpy.int8)

    def nActions(self):
        return len(THROTTLE)

    def dState(self):
        return nFRAMES, len(self.vehicles), 3

    def sampleAction(self):
        return numpy.random.randint(self.nActions())

    def locations(self):
        base = numpy.array([self.av.get_location().x, self.av.get_location().y, self.av.get_transform().rotation.yaw])
        location = [[vehicle.get_location().x, vehicle.get_location().y, vehicle.get_transform().rotation.yaw] for vehicle in self.vehicles]
        location.append(self.destination)
        return (numpy.array(location) - base).astype(numpy.int8)

    def prettifyState(self, rawState):
        F, V, D = self.dState()
        print(rawState.shape)
        return rawState.reshape(-1, F, V, D)

    def destroy(self):
        if self.collisionSensor is not None:
            self.collisionSensor.destroy()
        if self.gridSensor is not None:
            self.gridSensor.destroy()

        for actor in self.world.get_actors():
            if 'vehicle' in actor.type_id:
                actor.destroy()

        self.av = None
        self.gridSensor = None
        self.collisionSensor = None
        self.vehicles = []

    def __del__(self):
        self.destroy()
