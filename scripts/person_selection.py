"""Selection remains attached to a temporary ID, not detector order."""


class PersonSelection:
    def __init__(self):
        self.track_id = None

    def update(self, rows, key):
        if self.track_id not in [r["track_id"] for r in rows]:
            self.track_id = None
        if ord("1") <= key <= ord("6"):
            index = key - ord("1")
            self.track_id = rows[index]["track_id"] if index < len(rows) else None
        return self.track_id
