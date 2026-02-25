from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

ea = EventAccumulator("runs/unet_training_dict")
ea.Reload()

train_loss = [(e.step, e.value) for e in ea.Scalars("train_loss")]
val_dice = [(e.step, e.value) for e in ea.Scalars("val_mean_dice")]

steps, losses = zip(*train_loss)
vsteps, vdice = zip(*val_dice)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(steps, losses)
ax1.set_title("Train Loss")
ax1.set_xlabel("Epoch")

ax2.plot(vsteps, vdice)
ax2.set_title("Validation Dice")
ax2.set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("training_curves.png")
print("Saved to training_curves.png")
