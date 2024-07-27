class MusicStructureService:
    def generate_structure(self, genre, total_length):
        structures = {
            'pop': ['intro', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro'],
            'jazz': ['intro', 'theme', 'solo', 'theme', 'solo', 'theme', 'outro'],
            'classical': ['exposition', 'development', 'recapitulation']
        }

        chosen_structure = structures.get(genre, structures['pop'])
        section_length = total_length / len(chosen_structure)

        structured_composition = []
        for section in chosen_structure:
            structured_composition.append({
                'name': section,
                'start_time': len(structured_composition) * section_length,
                'end_time': (len(structured_composition) + 1) * section_length
            })

        return structured_composition