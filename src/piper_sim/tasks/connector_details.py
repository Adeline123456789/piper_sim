"""Photo-derived geometry for the recorded plug station. Local dimensions are metres."""
import xml.etree.ElementTree as ET

import numpy as np
from shapely import LineString, Polygon, box as rect, constrained_delaunay_triangles
from shapely.affinity import translate

from ..config import ROOT
from .connector import prism, rounded_rectangle
from .mujoco import nums


IVORY = (.86, .86, .83, 1)
BLUE = (.39, .51, .73, 1)
YELLOW = (.85, .65, .035, 1)
GOLD = (.35, .25, .11, 1)
DARK = (.026, .026, .028, 1)
VISUAL = dict(contype="0", conaffinity="0")
CONTACT = dict(priority="1", solref=".0042 1", solimp=".98 .999 .0005")


def geom(body, kind, size, pos, color, **attrs):
    return ET.SubElement(body, "geom", type=kind, size=nums(size), pos=nums(pos),
                         rgba=nums(color), **attrs)


def solid(asset, body, name, points, faces, color, **attrs):
    ET.SubElement(asset, "mesh", name=name, vertex=nums(points), face=nums(faces), inertia="shell")
    ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, rgba=nums(color), **attrs)


def extrude(asset, body, name, polygon, low, high, color, **attrs):
    # Split the face into convex pieces so MuJoCo cannot fill an opening with a hull.
    for i, tri in enumerate(constrained_delaunay_triangles(polygon).geoms):
        pts, faces = prism(np.asarray(tri.exterior.coords)[:3], low, high)
        solid(asset, body, f"{name}/{i}", pts, faces, color, **attrs)


def loft(rings):
    n = len(rings[0])
    faces = [(0, i+1, i) for i in range(1, n-1)]
    for layer in range(len(rings)-1):
        for i in range(n):
            a = layer*n+i
            b = layer*n+(i+1) % n
            faces.extend([(a,b,b+n), (a,b+n,a+n)])
    top = (len(rings)-1)*n
    faces.extend((top,top+i,top+i+1) for i in range(1,n-1))
    return np.concatenate(rings), np.asarray(faces)


def outline(w, d, r):
    return np.asarray(rounded_rectangle(w, d, r))


def ring(xy, z):
    return np.column_stack((xy, np.full(len(xy), z)))


def ellipse(rx, ry):
    a = np.linspace(0, 2*np.pi, 48, endpoint=False)
    return np.column_stack([rx*np.cos(a), ry*np.sin(a)])


def decal(body, name, halfsize, pos, **attrs):
    return geom(body, "box", [*halfsize, .000015], pos, (1,1,1,1),
                material=name, **VISUAL, **attrs)


def dress_gray_holder(root, fixture):
    board = root.find(".//body[@name='task/board']")
    board.find("geom[@name='task/board_visual']").set("rgba", "0 0 0 0")
    # The original CAD collision pieces keep the through holes. The printed exterior
    # has a continuous silver skin and a darker throat immediately below each lip.
    exterior = rect(-.085,-.030,.085,.030)
    for aperture in fixture.assets["holes"]:
        lo, hi = np.array(aperture["bounds_xy_m"])
        exterior = exterior.difference(rect(*lo,*hi))
        x,y = (lo+hi)/2
        dx,dy = (hi-lo)/2
        for sign in (-1,1):
            geom(board,"box",[.0001,dy,.0128],[x+sign*(dx+.0001),y,.0201],DARK,**VISUAL)
            geom(board,"box",[dx,.0001,.0128],[x,y+sign*(dy+.0001),.0201],DARK,**VISUAL)
    for side in (-1, 1):
        geom(board,"box",[.085,.00075,.01645],[0,side*.02925,.01645],(.57,.57,.57,1))
        geom(board,"box",[.00075,.0285,.01645],[side*.08425,0,.01645],(.57,.57,.57,1))
    geom(board,"box",[.083,.028,.0001],[0,0,.0052],DARK,**VISUAL)
    vertices, faces = [], []
    for tri in constrained_delaunay_triangles(exterior).geoms:
        pts, indices = prism(np.array(tri.exterior.coords)[:3],.0329,.033)
        faces.extend(indices+len(vertices))
        vertices.extend(pts)
    pts = np.array(vertices)
    uv = (pts[:,:2] / [.170,.060] + .5) * [10,4]
    ET.SubElement(root.find("asset"),"mesh",name="holder/silver_top",vertex=nums(pts),
                  face=nums(faces),texcoord=nums(uv),inertia="shell")
    ET.SubElement(board,"geom",type="mesh",mesh="holder/silver_top",material="silver_grain",**VISUAL)


def dress_plugs(asset, world, task):
    yellow = world.find("body[@name='task/yellow_plug']")
    spec = task.config["yellow_plug"]
    radius, height = spec["face_radius_m"], spec["housing_height_m"]
    yellow.remove(yellow.find("geom[@name='task/yellow_plug/handle']"))
    yellow.find("geom[@name='task/yellow_plug/rim']").set("rgba", nums(YELLOW))
    # Circular flange becomes a flattened grip with rounded shoulders and a hollow cable entry.
    # In the assembled end view the wide ribbed face looks down the board at the empty slot.
    edge = Polygon(outline(.024, .030, .0085)).boundary
    rim = ellipse(radius, radius)
    neck = np.array([edge.intersection(LineString([(0,0),v])).coords[0] for v in rim])
    offset = np.array([.002, 0.])  # the cable entry sits behind the screw boss
    shoulder = .62*rim + .38*neck
    rings = [ring(rim, .007), ring(rim*.97, .010), ring(shoulder, .015),
             ring(neck*1.15 + .65*offset, .023), ring(neck*1.04 + offset, .035),
             ring(neck + offset, height-.007)]
    vertices, faces = loft(rings)
    solid(asset,yellow,"plug/grip",vertices,faces,YELLOW,friction=".7 .003 .0001")
    # The rounded lip surrounds a recessed cable opening. A ring mesh leaves it visible.
    opening = ellipse(.0060,.0058) + offset
    layers = [ring(neck+offset,height-.007),ring(neck*.97+offset,height-.0018),
              ring(neck*.87+offset,height),ring(opening,height),ring(opening,height-.007)]
    faces = []
    for start in range(0, (len(layers)-1)*len(neck), len(neck)):
        for i in range(len(neck)):
            a, b = start+i, start+(i+1)%len(neck)
            faces.extend([(a,b,b+len(neck)),(a,b+len(neck),a+len(neck))])
    solid(asset,yellow,"plug/entry_rim",np.concatenate(layers),faces,YELLOW,**VISUAL)
    # Closed convex pieces provide the corresponding fingertip contact around the entry.
    cap = Polygon(neck+offset).difference(Polygon(opening))
    extrude(asset,yellow,"plug/entry_contact",cap,height-.007,height,(0,0,0,0),group="3")
    geom(yellow,"cylinder",[.006,.0001],[*offset,height-.0069],DARK,**VISUAL)
    geom(yellow,"cylinder",[.002,.0001],[*offset,height-.0067],GOLD,**VISUAL)
    # Side screw on the molded boss, plus raised traction ribs on the flat face.
    for name, rad, x, depth, color in [
        ("boss", .004, -.0148, .0028, YELLOW), ("screw", .0026, -.0178, .00045, GOLD)
    ]:
        geom(yellow,"cylinder",[rad,depth],[x,0,.016],color, name="plug/"+name,
             quat=".70710678 0 -.70710678 0", **VISUAL)
    for width, height in [(.00155,.00028),(.00028,.00155)]:
        geom(yellow,"box",[.0001,width,height],[-.01825,0,.016],DARK,**VISUAL)
    for z in (.030,.034,.038,.042):
        geom(yellow,"capsule",[.00045],[0,0,0],YELLOW,**VISUAL,
             fromto=nums([-.0109,-.006,z,-.0109,.006,z])).attrib.pop("pos")

    black = world.find("body[@name='task/black_charger']")
    w,d,h = task.config["black_charger"]["housing_size_m"]
    # The USB-A receptacle is recessed into the charger's top.
    housing = asset.find("mesh[@name='task/black_charger/housing']")
    points = np.fromstring(housing.get("vertex"),sep=" ").reshape(-1,3)
    points[points[:,2] > 0,2] = h-.004
    housing.set("vertex",nums(points))
    boundary = Polygon(outline(w,d,.004))
    opening = rect(-.0066,-.00265,.0066,.00265)
    lip = boundary.difference(Polygon(outline(w-.002,d-.002,.0035)))
    extrude(asset,black,"charger/top_rim",lip,h-.004,h,(.041,.039,.037,1))
    inner = Polygon(outline(w-.002,d-.002,.0035)).difference(opening)
    extrude(asset,black,"charger/top_plate",inner,h-.004,h-.00035,(.054,.052,.050,1))
    geom(black,"box",[.0065,.0026,.00008],[0,0,h-.0038],DARK,**VISUAL)
    geom(black,"box",[.0058,.00065,.00035],[0,-.0013,h-.00175],(.78,.76,.68,1),**VISUAL)
    decal(black,"charger_print",[h*.43,d*.42],[-w/2-.00005,0,h*.50],
          quat=".70710678 0 -.70710678 0")


def strip_geometry(asset, strip, task):
    length, width, total_h = task.config["strip_size_m"]
    h = task.socket_face_z
    # The collar and mounting ear at +X fit inside the measured 212 mm span.
    shell_l = length-.004
    def shell(w, d, r):
        return outline(w,d-.000035,r) + [-.002,.0000175]
    outside = Polygon(shell(shell_l,width,.005))
    inner = Polygon(shell(shell_l-.004,width-.004,.004))
    walls = outside.difference(inner)
    extrude(asset,strip,"strip/base",outside,0,.002,IVORY)
    extrude(asset,strip,"strip/lower",walls,.002,h*.53,IVORY)
    extrude(asset,strip,"strip/seam",walls,h*.53,h*.53+.00055,(.62,.62,.60,1))
    extrude(asset,strip,"strip/upper",walls,h*.53+.00055,h-.0025,IVORY)
    top = outside
    tile = Polygon(outline(.036,.037,.0023))  # centred 6 mm above the angled pin pair
    holes = []
    for part in task.fixture.parts["yellow_plug"]:
        if part["name"].startswith("pin_"):
            holes.append(Polygon(np.array(part["points"])[:4,:2]).buffer(task.config["slot_clearance_m"],join_style="mitre"))
    holes.extend(rect(x-.00135,.009,x+.00135,.017) for x in (-.0088,.0088))
    for i,(x,y) in enumerate(task.outlets):
        panel = translate(tile,x,y+.006)
        top = top.difference(panel)
        cuts = ([rect(dx-.0028,-.006,dx+.0028,.0065) for dx in (-.0048,.0048)]
                if i==0 else holes)
        for hole in cuts:
            panel = panel.difference(translate(hole,x,y+(.006 if i==0 else 0)))
        extrude(asset,strip,f"strip/panel_{i}",panel,h-.0025,h,BLUE,**CONTACT)
        # Real USB tongues catch the light below the openings.
        if i==0:
            for dx in (-.0048,.0048):
                geom(strip,"box",[.00065,.0057,.001],[x+dx-.0013,y+.006,h-.004],
                     (.50,.49,.42,1),**VISUAL)
        # Slightly bevelled raised rocker on a dark recessed border.
        by = task.config["button_y_m"]
        def keyring(w,d,z):
            return ring(outline(w,d,.0028)+[x,by],z)
        pts,faces = loft([keyring(.022,.014,h-.0001),keyring(.022,.014,h+.0003)])
        solid(asset,strip,f"strip/key_seam_{i}",pts,faces,(.33,.36,.40,1))
        pts,faces = loft([keyring(.020,.0125,h+.0003),keyring(.0197,.0122,total_h-.001),
                          keyring(.018,.0108,total_h-.00012)])
        solid(asset,strip,f"strip/key_{i}",pts,faces,BLUE)
        angles = np.linspace(np.deg2rad(115),np.deg2rad(425),11)
        points = np.column_stack([x+.00165*np.cos(angles),by+.00165*np.sin(angles)])
        for a,b in zip(points[:-1],points[1:]):
            ET.SubElement(strip,"geom",type="capsule",size=".00010",
                          fromto=nums([*a,total_h-.00010,*b,total_h-.00010]),
                          rgba=".26 .36 .55 1",**VISUAL)
        geom(strip,"box",[.00012,.00095,.00005],[x,by+.001,total_h-.00005],(.26,.36,.55,1),**VISUAL)
        geom(strip,"sphere",[.0011],[x-.013,by-.009,h-.0007],(.34,.09,.08,1),**VISUAL)
    extrude(asset,strip,"strip/face",top,h-.0025,h,IVORY,**CONTACT)
    # Dark inner backing is below the longest prong; it makes actual open slots legible.
    geom(strip,"box",[shell_l/2-.004,width/2-.004,.00015],[-.002,0,.0023],DARK,**VISUAL)
    decal(strip,"brand",[.010,.0047],[.093,0,h+.00002],quat=".70710678 0 0 -.70710678")
    decal(strip,"qc",[.015,.009],[.021,-width/2+.000015,h*.46],quat=".70710678 .70710678 0 0")
    end = length/2
    for rad,x,half,color in [(.0068,end-.0035,.0005,IVORY),(.0058,end-.0024,.0016,IVORY),
                             (.0044,end-.00075,.00012,(.47,.43,.23,1))]:
        geom(strip,"cylinder",[rad,half],[x,.004,.012],color,quat=".70710678 0 .70710678 0")
    ear = Polygon(outline(.008,.014,.002)+[end-.004,-.023])
    opening = Polygon(ellipse(.0015,.003)+[end-.003,-.023])
    extrude(asset,strip,"strip/ear",ear.difference(opening),.0002,.0025,IVORY)


def add_connector_details(root, strip, task):
    asset,world = root.find("asset"),root.find("worldbody")
    for name in ("silver_grain","qc","brand","charger_print"):
        path = ROOT/"assets/tasks/connector_photo"/(name+".png")
        ET.SubElement(asset,"texture",name=name,type="2d",file=str(path))
        ET.SubElement(asset,"material",name=name,texture=name,
                      texrepeat="1 1",specular=".10")
    dress_gray_holder(root, task.fixture)
    dress_plugs(asset,world,task)
    strip_geometry(asset,strip,task)
    # Viewer inspection cameras; policy images still use the three RealSense streams.
    board = world.find("body[@name='task/board']")
    for body, name, pos in [(strip, "white_strip_detail", [0,-.223,.130]),
                             (board, "gray_holder_detail", [0,-.215,.157])]:
        ET.SubElement(body, "camera", name=name, pos=nums(pos), xyaxes="1 0 0 0 .45 .893")
    ET.SubElement(board, "camera", name="assembly_side", pos="-.072 -.244 .137",
                  xyaxes=".959 -.283 0 .093 .315 .944", fovy="34")
    ET.SubElement(board, "camera", name="assembly_end", pos="-.252 -.085 .155",
                  xyaxes=".320 -.947 0 .407 .138 .903", fovy="32")
