"""MuJoCo reconstruction of the photographed connector workstation (metres)."""
import xml.etree.ElementTree as ET

import numpy as np
from shapely import LineString, Point, Polygon, constrained_delaunay_triangles

from ..config import ROOT
from .connector import prism
from .connector_details import VISUAL, geom, outline
from .mujoco import nums

BLACK = (.035, .038, .043, 1)
SILVER = (.60, .62, .64, 1)
WHITE = (.84, .83, .80, 1)


def block(body, pos, extent, color=BLACK):
    return geom(body, "box", np.asarray(extent)/2, pos, color, **VISUAL)


def cable(body, points, radius, color=BLACK):
    for a, b in zip(points, points[1:]):
        ET.SubElement(body, "geom", type="capsule", size=str(radius),
                      fromto=nums([*a, *b]), rgba=nums(color), **VISUAL)


def desk_clamp(body, y, top):
    # Cast C-frame wraps around the operator-facing edge; silver pad meets the underside.
    cable(body, [(-.032,y,top+.004), (-.056,y,top+.004), (-.066,y,top-.008),
                 (-.066,y,-.059), (-.060,y,-.068), (-.035,y,-.068)], .0055)
    cable(body, [(-.035,y,-.122), (-.035,y,-.005)], .0027, SILVER)
    geom(body, "cylinder", [.009,.002], [-.035,y,-.002], BLACK, **VISUAL)
    cable(body, [(-.051,y,-.12), (-.019,y,-.12)], .0024)


def mounting_plate(asset):
    # One shared mesh, with open slots and through holes visible down to the desk.
    face = Polygon(outline(.116, .255, .008))
    for x in (-.031, .031):
        face = face.difference(LineString([(x,-.093),(x,.093)]).buffer(.0034,quad_segs=4))
    for x in (-.044, .044):
        for y in (-.113, .113):
            face = face.difference(Point(x,y).buffer(.0048,quad_segs=4))
    vertices, faces = [], []
    for tri in constrained_delaunay_triangles(face).geoms:
        points, indices = prism(np.asarray(tri.exterior.coords)[:3], 0, .011)
        faces.extend(indices + len(vertices))
        vertices.extend(points)
    ET.SubElement(asset,"mesh",name="station/mount_plate",vertex=nums(vertices),
                  face=nums(faces),inertia="shell")


def add_arm_mounts(asset, world, task):
    mounting_plate(asset)
    for side, sign in (("left",1), ("right",-1)):
        base = world.find(f"body[@name='{side}/base_link']")
        pos = np.fromstring(base.get("pos"), sep=" ")
        pos[1] += sign * task.config.get("arm_outward_offset_m", 0.)
        base.set("pos", nums(pos))
        mount = ET.SubElement(world,"body",name=side+"/mount_plate",pos=nums([pos[0],pos[1],0]))
        ET.SubElement(mount,"geom",type="mesh",mesh="station/mount_plate",pos=nums([.005,0,task.top]),
                      rgba=nums(BLACK), **VISUAL)
        for y in (-.085,.085):
            desk_clamp(mount, y, pos[2])
        for link in ("link2", "link3"):
            shell = base.find(f".//geom[@name='{side}/{link}/mesh0']")
            shell.set("rgba", ".35 .36 .37 1")
        # The walkthrough shows the camera lead tied to the forearm with slack at the elbow.
        forearm = base.find(f".//body[@name='{side}/link3']")
        cable(forearm,[(-.022,-.194,.013),(-.023,-.162,.022),
                       (-.023,-.094,.023),(-.019,-.054,.030),
                       (-.014,-.026,.042),(.004,.008,.044),(.019,.015,.037)],.0026)
        for y in (-.142,-.086):
            cable(forearm,[(-.039,y,.013),(-.036,y,.023),(-.010,y,.023),
                           (-.006,y,.015)],.0015)
        wrist = world.find(f".//body[@name='{side}/link6']")
        # Short USB lead loops behind the wrist bracket and moves with link 6.
        block(wrist, [-.077,-.078,.032], [.013,.013,.015], (.25,.25,.29,1))
        cable(wrist,[(-.077,-.083,.032),(-.075,-.085,.014),(-.061,-.079,-.009),
                     (-.043,-.059,-.024),(-.029,-.033,-.030)], .0024)
        geom(wrist,"cylinder",[.0292,.005],[-.0005,-.0005,.020],(.79,.53,.09,1),**VISUAL)
    for y in (-.073,.073):
        desk_clamp(world,y,task.top+.011)
    cable(world,[(-.042,.004,1.035),(-.044,.004,.73),(-.045,.005,.36),
                 (-.055,.005,.047),(-.064,.006,-.06),(-.052,.016,-.20)], .0024)


def rear_facing_panel(asset, body, name, material, width, height, pos):
    # Texture coordinates are as seen from the operator's side (-X).
    y, z = width/2, height/2
    ET.SubElement(asset,"mesh",name=name,inertia="shell",
                  vertex=nums([[0,y,-z],[0,-y,-z],[0,-y,z],[0,y,z]]),
                  texcoord="0 1 1 1 1 0 0 0",face="0 1 2 0 2 3")
    ET.SubElement(body,"geom",type="mesh",mesh=name,pos=nums(pos),material=material,**VISUAL)


def office_desk(world, name, xmin, xmax, ymin, ymax):
    desk = ET.SubElement(world, "body", name="office/"+name)
    block(desk, [(xmin+xmax)/2,(ymin+ymax)/2,.0125], [xmax-xmin,ymax-ymin,.025], WHITE)
    for x in (xmin+.075,xmax-.075):
        for y in (ymin+.070,ymax-.070):
            block(desk, [x,y,-.36], [.05,.05,.72], WHITE)
    return desk


def monitor(asset, desks):
    # Group by the tabletop supporting each stand. Coordinates are in the workcell frame.
    displays = [
        ("opposite", .955, .025, 0., "monitor_back", "flat"),
        ("huawei", 1.07, .94, .06, "huawei_back", "flat"),
        ("right", .842, -.842, -1.37, "small_monitor", "riser"),
        ("right", .335, -.862, -1.54, "lenovo_back", "round"),
        ("left", .36, 1.11, np.pi, None, "silver"),
        ("left", .36, 1.64, np.pi, None, "flat"),
        ("huawei", 1.10, 1.455, .14, "small_monitor", "flat"),
    ]
    for i, (desk,x,y,yaw,material,stand) in enumerate(displays):
        m = ET.SubElement(desks[desk],"body",name=f"office/monitor_{i}",pos=nums([x,y,0]),
                          quat=nums([np.cos(yaw/2),0,0,np.sin(yaw/2)]))
        riser = .042 if stand == "riser" else 0.
        if riser:
            geom(m,"cylinder",[.117,riser/2],[0,0,.025+riser/2],(.45,.075,.065,1),**VISUAL)
        color = SILVER if stand == "silver" else BLACK
        if stand in ("round","riser"):
            geom(m,"cylinder",[.096,.005],[0,0,.030+riser],BLACK,**VISUAL)
        else:
            block(m, [0,0,.034], [.17,.24,.018],color)
        block(m, [.014,0,.146+riser], [.032,.038,.23],color)
        block(m, [.044,0,.324+riser], [.025,.512,.296],color)
        if material:
            rear_facing_panel(asset,m,f"display/back_{i}",material,.506,.292,[.030,0,.324+riser])
        # The two screens on the operator's left show their working faces in the final pan.
        front = ET.SubElement(m,"body",pos=nums([.058,0,.324+riser]),quat="0 0 0 1")
        rear_facing_panel(asset,front,f"display/screen_{i}","office_screen",.498,.281,[0,0,0])
        cable(m,[(.028,.07,.193+riser),(.013,.068,.14),(-.018,.02,.047),
                 (-.065,.032,.035+riser)],.002)


def add_workstation(root, task):
    asset, world = root.find("asset"), root.find("worldbody")
    background = task.config.get("workstation_background", "recording")
    if background not in ("photos", "recording"):
        raise ValueError("workstation_background must be photos or recording")
    materials = ["desktop"]
    if background == "photos":
        materials.extend(["office_floor", "monitor_back", "tower_front", "office_glass", "small_monitor",
                          "lenovo_back", "huawei_back", "tissue_front", "office_screen"])
    for name in materials:
        path = ROOT/"assets/tasks/connector_photo"/(name+".png")
        ET.SubElement(asset,"texture",name=name,type="2d",file=str(path))
        ET.SubElement(asset,"material",name=name,texture=name,specular=".06",
                      texrepeat="12 12" if name == "office_floor" else "1 1")
    # Use the CAD collision surface and give the visible desktop radiused edges and UVs.
    table = world.find("geom[@name='table_collision']")
    center, half = (np.fromstring(table.get(k),sep=" ") for k in ("pos","size"))
    xy = outline(half[0]*2,half[1]*2,.014)
    pts, faces = prism(xy,-half[2],half[2])
    ET.SubElement(asset,"mesh",name="station/table",vertex=nums(pts),face=nums(faces),
                  texcoord=nums(pts[:,:2]/[.65,.70]+.5),inertia="shell")
    original = world.find("geom[@name='cad_static_0']")
    original.attrib.update(mesh="station/table",material="desktop",pos=nums(center),rgba="1 1 1 1")
    add_arm_mounts(asset,world,task)
    for y in (-.61,.61):
        block(world,[.29,y,-.365],[.082,.067,.70],WHITE)
        block(world,[.29,y,-.707],[.57,.084,.042],WHITE)
    block(world,[.29,0,-.039],[.10,1.22,.074],WHITE)
    block(world,[-.047,.57,-.020],[.052,.148,.034])
    for y in (.527,.548,.569,.590):
        geom(world,"sphere",[.0024],[-.074,y,-.018],SILVER,**VISUAL)
    # Front view from the operator's side. Inspection cameras are outside the policy rig.
    ET.SubElement(world,"camera",name="station_front",pos="-1.28 .13 .56",
                  xyaxes="-.079 -.997 0 .378 -.030 .925",fovy="47")
    ET.SubElement(world,"camera",name="workbench_walkthrough",pos="-.68 .04 .57",
                  xyaxes="-.045 -.999 0 .317 -.014 .948",fovy="48")
    if background == "photos":
        add_office(asset, world)


def add_office(asset, world):
    world.find("geom[@name='ground']").attrib.update(pos="0 0 -.73",size="5 5 .05",
                                                    material="office_floor",rgba="1 1 1 1")
    # Adjacent office table, monitors and blue glazed divider visible behind the arms.
    desks = {
        "opposite": office_desk(world,"opposite",.652,1.345,-.698,.698),
        "right": office_desk(world,"right",-.054,1.345,-1.404,-.705),
        "left": office_desk(world,"left",-.055,.645,.705,2.105),
        "huawei": office_desk(world,"huawei",.652,1.345,.705,2.105),
    }
    monitor(asset,desks)
    # Keyboard in front of the black display, parallel to the long desk edge.
    block(desks["left"],[.145,1.64,.036],[.13,.36,.022])
    for row in range(5):
        for col in range(16):
            block(desks["left"],[.107+.019*row,1.4975+.019*col,.049],
                  [.016,.016,.004],(.10,.11,.12,1))
    # Closed laptop and USB hub at the back edge; storage bin is open at the top.
    block(world,[.771,.264,.036],[.175,.239,.020],(.105,.108,.115,1))
    block(world,[.769,-.114,.042],[.093,.038,.031],(.18,.18,.20,1))
    for x in (.708,.808):
        block(world,[x,.58,.073],[.005,.151,.094],(.67,.68,.68,1))
    for y in (.507,.653):
        block(world,[.758,y,.073],[.099,.005,.094],(.67,.68,.68,1))
    block(world,[.758,.58,.029],[.10,.15,.006],WHITE)
    # Tea glass with amber liquid and spoon seen beside the tissue box.
    geom(world,"cylinder",[.034,.008],[.79,-.202,.034],(.56,.31,.085,1),**VISUAL)
    geom(world,"cylinder",[.035,.036],[.79,-.202,.071],(.84,.88,.85,.24),**VISUAL)
    cable(world,[(.795,-.202,.047),(.786,-.181,.133)],.0022,(.54,.36,.19,1))
    block(world,[1.50,-.06,.02],[.25,1.39,.73],(.75,.75,.73,1))
    for y in (-.52,-.06,.40):
        block(world,[1.37,y,.021],[.008,.455,.70],WHITE)

    block(world,[1.65,0,.68],[.025,4.2,1.55],(.21,.48,.65,1))
    for i, y in enumerate(np.arange(-2,2.1,.8)):
        rear_facing_panel(asset,world,f"office/pane_{i}","office_glass",.80,.82,[1.635,y,.56])
    for y in (-2.4,-1.6,-.8,0,.8,1.6,2.4):
        block(world,[1.63,y,.68],[.026,.028,1.55])
    block(desks["opposite"],[1.185,.025,.0375],[.105,.34,.025],(.49,.43,.66,1))
    for x in np.linspace(1.151,1.219,5):
        for y in np.linspace(-.121,.181,16):
            block(desks["opposite"],[x,y,.052],[.012,.016,.006],WHITE if y<.11 else (.62,.51,.79,1))
    block(world,[.748,-.34,.066],[.085,.14,.082],(.12,.38,.68,1))
    block(world,[.75,-.34,.108],[.035,.079,.006],WHITE)
    rear_facing_panel(asset,world,"office/tissue_label","tissue_front",.139,.080,[.705,-.34,.066])
    for y in (-.39,.36):
        block(world,[.29,y,-.50],[.36,.154,.451],(.055,.057,.067,1))
    rear_facing_panel(asset,world,"office/pc_face","tower_front",.154,.451,[.109,-.39,-.50])
    # Passive cable runs below the desk. Their shape is an estimate from the front photo.
    for y, direction in [(-.31,1),(.31,-1)]:
        cable(world,[(-.050,y,.035),(-.078,y,-.028),(-.095,y,-.24),
                     (-.105,y+.06*direction,-.47),(-.08,y+.18*direction,-.705),
                     (.08,y+.16*direction,-.721),(.16,y,-.715)],.003)
    # Two inline power supplies and their slack leads lie on the operator side.
    for x,y,direction in [(-.20,.13,1),(-.105,-.14,-1)]:
        block(world,[x,y,-.710],[.078,.152,.035],BLACK)
        cable(world,[(x,y+.075,-.71),(x-.016,y+.107,-.724),
                     (x-.11,y+.11,-.724),(x-.16,y+.01,-.724),
                     (x-.14,y-.10,-.724),(x-.045,y-.13,-.724),
                     (.04,y+direction*.19,-.724),(.11,y+direction*.22,-.69)],.0027)
