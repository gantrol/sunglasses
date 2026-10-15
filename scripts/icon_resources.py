"""Inspect PE icon resources as data; never load or start the executable."""

import struct


def verify_embedded_icon(executable, ico):
    import pefile

    if ico[:4] != b"\x00\x00\x01\x00":
        raise ValueError("Invalid source ICO header")
    count = struct.unpack_from("<H", ico, 4)[0]
    pe = pefile.PE(data=executable, fast_load=True)
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
        types = {entry.id: entry for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries}
        groups = {entry.id: entry for entry in types[14].directory.entries}
        group_entries = groups[101].directory.entries
        if len(group_entries) != 1:
            raise ValueError("Expected one neutral application icon group")
        def data(entry):
            info = entry.data.struct
            return pe.get_data(info.OffsetToData, info.Size)
        group = data(group_entries[0])
        if group[:6] != ico[:6] or len(group) != 6+14*count:
            raise ValueError("Embedded application icon sizes differ from the source ICO")
        icons = {entry.id: entry for entry in types[3].directory.entries}
        for index in range(count):
            source_entry = ico[6+16*index:22+16*index]
            embedded_entry = group[6+14*index:20+14*index]
            if source_entry[:12] != embedded_entry[:12]:
                raise ValueError("Embedded application icon metadata differs from the source ICO")
            size, offset = struct.unpack_from("<II", source_entry, 8)
            resource_id = struct.unpack_from("<H", embedded_entry, 12)[0]
            if data(icons[resource_id].directory.entries[0]) != ico[offset:offset+size]:
                raise ValueError("Embedded application icon pixels differ from the source ICO")
    except (AttributeError, KeyError, struct.error, IndexError) as error:
        raise ValueError("Executable is missing the application icon resource") from error
    finally:
        pe.close()
