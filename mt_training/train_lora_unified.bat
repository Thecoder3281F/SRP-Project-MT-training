@echo off
REM Activate virtualenv and run unified LoRA trainer
call "C:\Users\user\Desktop\srp\Scripts\Activate.ps1"
python train_lora_unified.py %*
