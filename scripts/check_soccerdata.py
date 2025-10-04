import soccerdata as sd
print("soccerdata", getattr(sd, "__version__", "?"))
print("FBref available leagues (first 10):", sd.FBref.available_leagues()[:10])
fb = sd.FBref(leagues=["ENG-Premier League"], seasons=[2024])
print("FBref schedule shape:", fb.read_schedule().shape)
