import numpy as np
from hopp.tools.layout.location_unit_conversion import vert_transform_lat_lon_to_m
from matplotlib import pyplot as plt
import shapely

if __name__ == "__main__":

    area_lat_long = np.loadtxt("input_files/resources/oahu_north_call_area_lat_lon.csv", delimiter=",")

    area_m = vert_transform_lat_lon_to_m(area_lat_long)

    x = [p[0] for p in area_m]
    y = [p[1] for p in area_m]

    polygon = shapely.geometry.Polygon(area_m)
    print(f"Full area: {polygon.area} m^2")
    print(f"Full n points: {len(x)}")

    simple_polygon = polygon.simplify(500, preserve_topology=False)
    print(f"Simple area: {simple_polygon.area} m^2")
    x_simple = simple_polygon.exterior.coords.xy[0]
    y_simple = simple_polygon.exterior.coords.xy[1]
    print(f"Simple n points: {len(x_simple)}")

    np.savetxt("oahu_north_call_area.csv", np.c_[x, y], header="easting (m), northing (m)")

    np.savetxt("oahu_north_call_area_simplified.csv", np.c_[x_simple, y_simple], header="easting (m), northing (m)")

    fig, ax = plt.subplots(1)
    ax.plot(x, y, label="original")
    ax.plot(x_simple, y_simple, "--", label="simple")
    ax.set(aspect="equal")
    plt.legend()
    plt.show()