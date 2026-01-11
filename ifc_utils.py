import ifcopenshell
import ifcopenshell.guid

def create_ifc_covering_type(ifc_file: ifcopenshell.file, type_name: str, description: str = None, element_type: str = None) -> ifcopenshell.entity_instance:
    """
    Creates an IfcCoveringType definition in the given ifcopenshell file.

    Args:
        ifc_file: The ifcopenshell file object.
        type_name: The name of the IfcCoveringType.
        description: An optional description for the type.
        element_type: An optional element type (e.g., 'FLOORING', 'ROOFING').

    Returns:
        The created IfcCoveringType entity instance.
    """
    # In a full model, you would pass a proper IfcOwnerHistory.
    # For this utility, we assume it's handled externally or not critical for type definition alone.
    owner_history = None

    covering_type = ifc_file.create_entity(
        "IfcCoveringType",
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=owner_history,
        Name=type_name,
        Description=description,
        ApplicableOccurrence=None,
        HasPropertySets=[],
        RepresentationMaps=[],
        Tag=None,
        ElementType=element_type
    )
    return covering_type

def create_ifc_covering_instance(
    ifc_file: ifcopenshell.file,
    covering_type: ifcopenshell.entity_instance,
    instance_name: str,
    description: str = None,
    predefined_type: str = None,
    object_placement=None,
    representation=None
) -> ifcopenshell.entity_instance:
    """
    Creates an IfcCovering instance and links it to an IfcCoveringType definition.

    Args:
        ifc_file: The ifcopenshell file object.
        covering_type: The IfcCoveringType entity instance to link to.
        instance_name: The name of the IfcCovering instance.
        description: An optional description for the instance.
        predefined_type: An optional predefined type (e.g., 'FLOORING', 'ROOFING').
        object_placement: An optional IfcObjectPlacement entity.
        representation: An optional IfcProductRepresentation entity.

    Returns:
        The created IfcCovering entity instance.
    """
    owner_history = None # Placeholder

    covering_instance = ifc_file.create_entity(
        "IfcCovering",
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=owner_history,
        Name=instance_name,
        Description=description,
        ObjectType=covering_type.Name, # Link to the type by name
        ObjectPlacement=object_placement,
        Representation=representation,
        Tag=None,
        PredefinedType=predefined_type
    )

    # Connect the instance to its type definition using IfcRelDefinesByType
    ifc_file.create_entity(
        "IfcRelDefinesByType",
        GlobalId=ifcopenshell.guid.new(),
        OwnerHistory=owner_history,
        Name=f"Defines {covering_instance.Name} by {covering_type.Name}",
        Description=None,
        RelatedObjects=[covering_instance],
        RelatingType=covering_type
    )
    return covering_instance
