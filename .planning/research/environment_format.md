# Environment file format: review and schema draft (T1.1)

Author: Pratham · Date: 23 Nov 2026 · Status: draft, to be signed off by Linda together with the solver decision

## 1. Recommendation of the plan: JSON with a pydantic schema

**Agreed.** The project already loads all scenarios as JSON through pydantic (`ScenarioConfig`), colors
already exist as `ColorValue`, and no new dependency is needed. The environment lives in its own file
(`environment_file`), or inline under `environment`, so one world can be reused across scenarios.

## 2. What other formats do

| Format | What it describes | Shapes | Colour / transparency |
|---|---|---|---|
| SDFormat (Gazebo, XML) | whole worlds: model, link, `<collision>` and `<visual>` | spec 1.11: box, capsule, cone, cylinder, ellipsoid, sphere, plane, mesh, heightmap, ... | `<material>` with RGBA vectors; `<transparency>`: 0 = opaque, 1 = fully transparent |
| URDF (ROS, XML) | one robot | box, cylinder, sphere, mesh * | `rgba` * |
| MJCF (MuJoCo, XML) | models and worlds | plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh, sdf (no cone) | `rgba`, default `0.5 0.5 0.5 1` |
| AirSim | JSON `settings.json` for vehicles and sensors | none: the environment is the Unreal scene | none |
| Flightmare | Unity scenes chosen by an ID; objects spawned from C++ code with a prefab ID | none in a file | none |

\* Not yet checked against the documentation.

Mesh formats (OBJ, STL, glTF) hold triangles only and do not say "this is a cylinder", so they do
not fit the solver and are left for a later stage. A custom description is therefore normal for drone
simulators. Our primitives (box, cylinder, sphere, cone) all exist in SDFormat 1.11, so an SDF importer
would be a pure mapping: `size` and `radius` map directly, `opacity` = 1 - `transparency`,
`position` plus `z` become the pose of the shape's centre.

## 3. Traps to avoid

- **Transparency is inverted in SDF** (0 = opaque). Our field is `opacity` (1 = opaque), because
  `alpha` is already used by `DroneConfig` for the adaptive safety zone.
- **Half or full sizes.** Our file uses full sizes (`size`, `height`); `ObstacleConfig` uses
  `half_extents`, so the loader converts. MuJoCo boxes use half-sizes, the SDF examples use full
  lengths (both to be confirmed against the specification).
- **Cylinders and cones stand along z**, as in SDF.

## 4. Schema draft (five example objects)

```json
{
  "objects": [
    {"id": "house-1", "kind": "house",    "position": [-4.0, 0.0], "size": [6.0, 4.0, 5.0],
     "yaw_deg": 90, "color": "lightgrey", "opacity": 0.8},
    {"id": "tower-1", "kind": "cylinder", "position": [3.0, 3.0], "radius": 0.6, "height": 6.0,
     "color": [0.5, 0.5, 0.5]},
    {"id": "fir-1",   "kind": "fir",      "position": [1.0, -2.0], "radius": 1.2, "height": 5.0,
     "trunk_radius": 0.2, "trunk_height": 1.0, "color": "#1f6f3a"},
    {"id": "ball-1",  "kind": "sphere",   "position": [0.0, 4.0], "radius": 0.8, "z": 3.0,
     "opacity": 0.5},
    {"id": "wall-1",  "kind": "box",      "position": [-1.0, 1.0], "size": [0.3, 5.0, 2.0]}
  ]
}
```

Units: metres and degrees. x and y are horizontal, z points up, the floor is `room.min[2]`.

| Field | Applies to | Type, unit | Default | Meaning and rule |
|---|---|---|---|---|
| `id` | all | text | `kind-N` | name of the object; duplicate ids are an error |
| `kind` | all | text | required | `box`, `house`, `cylinder`, `sphere`, `fir`; `composite` (explicit parts) later |
| `position` | all | `[x, y]`, m | required | centre of the footprint (not a corner) |
| `z` | all | float, m | floor | height of the object's lowest point |
| `size` | `box`, `house` | `[lx, ly, lz]`, m | required | full side lengths, each greater than 0 |
| `yaw_deg` | all (no effect on round kinds) | int, degrees | 0 | only 0, 90, 180, 270; swaps x and y of `size` |
| `radius` | `cylinder`, `sphere`, `fir` | float, m | required | cylinder, sphere: radius; fir: crown radius; greater than 0 |
| `height` | `cylinder`, `fir` | float, m | required | cylinder: full height; fir: total height (trunk plus crown); greater than 0 |
| `trunk_radius` | `fir` | float, m | constant, set in T1.2 | radius of the trunk |
| `trunk_height` | `fir` | float, m | constant (the spike uses 20 % of `height`) | height of the trunk |
| `color` | all | name, hex string or `[r, g, b]` | per-kind default (T2.1) | like `drone_color`; numbers 0 to 1 or 0 to 255 |
| `opacity` | all | float | 1.0 | 0 to 1, 1 = opaque; a value outside is an error |

On scenario level (T1.3): `environment_file` is a path, looked up relative to the scenario file, then in
`configs/environments/`, then as an absolute path. `environment` is the same list given inline. Objects
from the file come first, inline objects after. Objects must lie inside `room` if one is set, and no
drone start or target may lie inside an object.

Terms: **kind** = type of object. **Footprint** = the outline of the object on the floor. **Lowest
point** = the bottom of the object. **Part** = one simple shape of an object (a fir has two: trunk and
crown). **Primitive** = box, cylinder, sphere or cone.

The file is flat and per `kind`. Internally each object is expanded into parts (a fir becomes a
cylinder and a cone), which follows the SDF split of one object into shapes. The solver receives the
simplified primitives, chosen by the decision in `environment_shapes.md`.

## 5. Open points

- Plan T2.1 suggests per-kind default colours with an opacity of 0.8; here the field default is 1.0.
  To settle in T2.1.
- CSV import (`x,y,kind,radius,height`) only if T1.3 finds it useful.
