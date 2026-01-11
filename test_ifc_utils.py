import ifcopenshell
import os
import unittest
# Assuming ifc_utils.py is in the same directory or accessible via sys.path
from ifc_utils import create_ifc_covering_type, create_ifc_covering_instance

class TestIfcCoveringCreation(unittest.TestCase):

    def setUp(self):
        # Create a new empty IFC file for each test
        self.ifc_file = ifcopenshell.file()
        # No need to create owner history directly if it's handled by ifcopenshell.guid.new()
        # or other defaults. IfcOpenShell will often create a default owner history if none is provided.

    def test_create_ifc_covering_type(self):
        type_name = "MyFloorTileType"
        description = "Standard ceramic floor tiles"
        element_type = "FLOORING"

        covering_type = create_ifc_covering_type(self.ifc_file, type_name, description, element_type)

        self.assertIsNotNone(covering_type)
        self.assertEqual(covering_type.is_a(), "IfcCoveringType")
        self.assertEqual(covering_type.Name, type_name)
        self.assertEqual(covering_type.Description, description)
        self.assertEqual(covering_type.ElementType, element_type)

        # Verify it can be found in the file
        found_type = self.ifc_file.by_guid(covering_type.GlobalId)
        self.assertEqual(found_type, covering_type)

    def test_create_ifc_covering_instance(self):
        type_name = "WallPaintType"
        instance_name = "LivingRoomWallPaint"
        predefined_type = "CLADDING"
        type_description = "Acrylic wall paint"
        instance_description = "Paint for living room walls"

        # Create the type first
        covering_type = create_ifc_covering_type(self.ifc_file, type_name, type_description)

        # Create dummy placement and representation (minimal for test, real ones would be more complex)
        # OwnerHistory is often required for IfcProduct and its subtypes like IfcCovering
        owner_history = self.ifc_file.create_entity("IfcOwnerHistory") # Minimal owner history
        
        # Location for IfcAxis2Placement3D
        location = self.ifc_file.create_entity("IfcCartesianPoint", (0., 0., 0.))
        axis2placement3d = self.ifc_file.create_entity("IfcAxis2Placement3D", location)
        local_placement = self.ifc_file.create_entity("IfcLocalPlacement", RelativePlacement=axis2placement3d)
        
        # Minimal representation for testing
        # ContextOfItems is usually an IfcGeometricRepresentationContext. For minimal, we can set it to None
        # Create a minimal IfcGeometricRepresentationContext
        geometric_context = self.ifc_file.create_entity(
            "IfcGeometricRepresentationContext",
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=1.0E-05,
            WorldCoordinateSystem=axis2placement3d # Use the same axis system
        )
        # Create a minimal item for the representation (e.g., a single point for geometry)
        geometric_curve_set = self.ifc_file.create_entity(
            "IfcGeometricCurveSet",
            Elements=[self.ifc_file.create_entity("IfcCartesianPoint", (0., 0., 0.))]
        )
        
        shape_representation = self.ifc_file.create_entity(
            "IfcShapeRepresentation", 
            ContextOfItems=geometric_context, 
            RepresentationIdentifier="Body", 
            RepresentationType="GeometricCurveSet", # Changed type to match the item
            Items=[geometric_curve_set] 
        )
        product_representation = self.ifc_file.create_entity("IfcProductRepresentation", Name="Body", Description="Body geometry", Representations=[shape_representation])


        covering_instance = create_ifc_covering_instance(
            self.ifc_file, 
            covering_type, 
            instance_name, 
            instance_description, 
            predefined_type,
            object_placement=local_placement,
            representation=product_representation
        )

        self.assertIsNotNone(covering_instance)
        self.assertEqual(covering_instance.is_a(), "IfcCovering")
        self.assertEqual(covering_instance.Name, instance_name)
        self.assertEqual(covering_instance.Description, instance_description)
        self.assertEqual(covering_instance.PredefinedType, predefined_type)
        
        # Verify linkage to type
        # In IfcOpenShell, ObjectType on the instance links to the type's Name
        self.assertEqual(covering_instance.ObjectType, covering_type.Name)

        # Check IfcRelDefinesByType by searching all IfcRelDefinesByType in the file
        found_rel_defines_by_type = None
        for rel_def in self.ifc_file.by_type("IfcRelDefinesByType"):
            if rel_def.RelatingType == covering_type and covering_instance in rel_def.RelatedObjects:
                found_rel_def = rel_def
                break
        
        self.assertIsNotNone(found_rel_def)
        self.assertEqual(found_rel_def.RelatingType, covering_type)
        self.assertIn(covering_instance, found_rel_def.RelatedObjects)


        # Verify it can be found in the file
        found_instance = self.ifc_file.by_guid(covering_instance.GlobalId)
        self.assertEqual(found_instance, covering_instance)

if __name__ == '__main__':
    unittest.main()
