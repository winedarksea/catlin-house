# Project Design Goals and Guidelines
* The name for this project will be `ifcPlot`
* Design  around IFC files and interact with them mostly from python in Visual Studio Code. IFC should be used so that this could be loaded into other major programs easily, such as Revit or ArchCAD.
* Use ifcopenshell to process the IFC files, ifctester to help validate the structure, and Bonsai BIM for 3d visualization (some users might use Bonsai to modify IFC details, but primarily it is intended in the stack just for visualization). For 2d details (wall and floorplan), plan to use matplotlib where possible.
* Uses adjustText or textalloc for matplotlib labels to improve layout.
* One constraint is I want this to be LLM coding agent friendly, for example files should not be too large, and to have built in tests that can quickly check changes without needing to load the full structure model into context (ifctester and regular unittests). 
* Format this repository as a reusable library so others can use it for their own projects. Note that this described project will be included in the pypi uploaded version, as the default starting project for others to modify as desired.
* Designed with North American imperial building standards (such as 2x4s and 8' by 4' panels commmonly) but ideally would be usable in metric as well.
* One nice to have feature of lower importance would be a full BOM export of materials (for example, how many cubic yards of concrete, and how many 104-5/8" studs).
* One potential difficulty is that windows and doors will be added and moved around later, so we expect to have structural headers and so on modified onto existing wall definitions.
* May have sub-variations of assemblies which need to be supported, without a full duplication, for minor changes to assemblies.
* IFC groups defined for each of the main subcontractors: concrete, framing, hvac, and plumbing. Drywall and cladding both also as separate groups so they can easily been hidden to show interior details. Finally a furnishings group for any added furniture and furnishings.
* Have a way to properly organize the "notes" that go into the floorplans and assembly details, in a way those can be viewed as markdown and loaded into the detail figures that are created as well, such as using Markdown with frontmatter (applied_to, tags, etc).
* An important overall goal is that details can be updated in as few locations as possible (for example, I might decide to switch floor joists from 11 7/8" to 14" depth, and I only need to make this change in one location for a floor, and all details that show this update automatically from this reference).
* Plots of assembly details and floorplans will use more color than is traditionally used in architectural details, such as doing Wall Type Schedules by color (with load bearing wall families getting warm colors, and cool colors for non-load bearing wall types), but style will generally be minimalistic, with colors and details added only where it has high value.
* Figures scaled to print at 1/4" to 1' scale

# High-Level Description of Design
Four structures
1. The house
2. The attached porch (technically a separate structure as it is freestanding, separate foundation) and sunken garden (retaining walls attached to porch)
3. the detached garage
4. a breezeway linking the house and the garage

The house:
36' by 36' measured at the sheathing, designed so 8' by 4' plywood sheathing can be used without cuts.
It consists of 4 levels called the basement, the main floor (also called first floor), the second floor, and the attic level floor (also called the third floor).
The basement is poured concrete, 12" thick walls (might be changed to 10" later) with 9' height (clear height above slab and below floor) with 9" thick concrete slab for the ceiling.
Basement floor is 3.5" thick concrete over 6 mil or greater poly vapor barrier, over 4" of rigid XPS foam, over 6" of aggregate.
Basement floor has one recessed area for a curbless shower.
Beneath basement are footing drains for drainage and radon that drain to a sump pump outside the basement.
Basement is subdivided into a grid of four smaller squares, with a cross of middle, load bearing concrete walls at 18' oc on each axis. 

The main structure of the upper floors is two loading bearing outer walls with a center midline load bearing wall. Joists (11 7/8" thickness) of about 18' span between the load bearing exterior side walls and the center midline wall (joists run east to west). These load bearing walls run north to south. The east to west running exterior walls bear only the load of the end floor joist and bear only a small fraction of house load compared to the north to south running walls.

For the floor between the first and second story, trimjoists are used. For the upper levels above that, standard i-joists are used.

The first floor has 2x6 LSL framing for all three load bearing walls. The center midline load bearing wall remains 2x6s for the upper levels but switches to standard dimensional lumber after the first floor. The side load bearing walls switch to 2x4 LSL for the second story, and 2x4 dimensional for the attic level studs. The east to west exterior walls (running east to west) are 2x4 LSL on the first floor, and 2x4 dimensional for the upper floors.

Most insulation is exterior insulation, refer to the attached details for further information.

Almost all of the framing is 16" oc spacing, unless noted otherwise. Standing seam siding and furring strips of siding and roofing are also at 16" oc. The standing seam siding for both the walls and roof is white.

The first floor has 9' high walls, and the second floor 9' high walls as well.  The attic level has 5' high end walls (interior height) that run north to south, and a 11' (interior height) giving it a centerline gable peak. It is framed almost exactly the same as the floors below, using 11 7/8" I-joists. The centerline load bearing wall supports the structural ridge beam, which mounts i-joists on LSSR sloped hangers to the sides. The i-joists (running east to west) then bear on the top plates of the side load bearing walls (running north to south). The attic level has a small finished area, but is mostly unfinished, with the option to finish later.

The stair section is also important for the full house structure. For connecting the basement, first, and second floors, a set of U-shaped stairs is present. These occupy a 2d space of 7' (east to west length) and 9' 8" (north to south length) and consist of 16 risers with a standard 10" (plus 1" nosing) tread.
These stairs are located on the northern side of the house immediately to the west of the centerline load bearing wall. The 7' dimension starts from the outside edge of the centerline wall 2x6.

As a result of the opening for the stairs, a set of smaller load bearing walls is present. In the basement, there is a poured concrete wall (8" thick) immediately to the west of the opening for the stairs (wall runs north south along side of stairs), and on the floor immediately above this is a load bearing wood stud wall, bearing shorter floor joists for the second floor (shorter due to the stair opening not being spanned).

The garage is 24' by 24' (again, measured at sheathing, for minimal sheathing cuts when using 8' by 4' sheathing) and has an 8' height wood stud wall on 22" of above grade ICF. See attached garage detail for full design parameters. The garage is 12' north of the house, and the house's west wall is aligned with the garage's west wall.

One interior room that is well defined with unique details is the sauna and attached shower. This is located in the basement and has a small area of recessed slab for the shower (4" lower than main basement slab, but otherwise similar slab details). This located on the west side of the centerline wall in the basement, up against the south side wall, with a door to it through the centerline concrete wall.
