"""
Optional Trainer Integration for Attention Visualization

This file shows how to minimally integrate attention visualization into the trainer.
You can add this code to your trainer.py if you want visualization during training,
or use the standalone visualize_attention.py script instead.
"""

# Add to imports at top of trainer.py:
# from utils.visualizer import AttentionVisualizer, AttentionHook


# Add to HTRTrainer.__init__ method:
def setup_visualizer(self):
    """
    Optional: Setup attention visualizer if enabled in config.
    Add this method and call it in __init__.
    """
    if hasattr(self.config, 'visualization') and self.config.visualization.enabled:
        from utils.visualizer import AttentionVisualizer
        self.visualizer = AttentionVisualizer(
            save_dir=self.config.visualization.save_dir,
            enabled=True
        )
        print(f'Attention visualization enabled, saving to: {self.config.visualization.save_dir}')
    else:
        self.visualizer = None


# Add to HTRTrainer class as a new method:
def visualize_attention_samples(self, epoch, num_samples=5):
    """
    Optional: Visualize attention maps for sample predictions.
    Call this method in your training loop when you want to save visualizations.
    
    Args:
        epoch: Current epoch number
        num_samples: Number of samples to visualize
    """
    if self.visualizer is None:
        return
    
    from utils.visualizer import AttentionHook
    
    print(f'Generating attention visualizations at epoch {epoch}...')
    
    config = self.config
    device = config.device
    
    self.net.eval()
    
    # Use validation set for visualization
    dataset = self.loaders['val'].dataset
    
    for idx in range(min(num_samples, len(dataset))):
        img, transcription = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            with AttentionHook(self.net, self.visualizer) as viz:
                output = self.net(img_batch)
                
                if config.arch.head_type == 'both':
                    output = output[0]
                
                # Decode prediction
                tdec = output.argmax(1).cpu().numpy()
                prediction = self.decode(tdec, self.classes['i2c'])
                
                # Save visualization
                viz.visualize_and_save(
                    input_image=img_batch,
                    transcription=transcription.strip(),
                    prediction=prediction.strip(),
                    sample_id=f'epoch_{epoch}_sample_{idx}',
                    save_individual=True
                )
    
    self.net.train()
    print(f'  Visualizations saved to {self.visualizer.save_dir}')


# In the main training loop (at the end of __main__ section), add:
"""
if __name__ == '__main__':
    # ... existing code ...
    
    htr_trainer = HTRTrainer(config)
    
    # Optional: Setup visualizer
    # htr_trainer.setup_visualizer()
    
    cnt = 1
    print('Training Started!')
    htr_trainer.test(0, 'test')
    
    for epoch in range(1, max_epochs + 1):
        htr_trainer.train(epoch)
        htr_trainer.scheduler.step()

        # save and evaluate the current model
        if epoch % config.train.save_every_k_epochs == 0:
            htr_trainer.save(epoch)
            htr_trainer.test(epoch, 'val')
            htr_trainer.test(epoch, 'test')
            
            # Optional: Generate attention visualizations
            # if hasattr(config, 'visualization') and config.visualization.enabled:
            #     if epoch % config.visualization.save_every_k_epochs == 0:
            #         htr_trainer.visualize_attention_samples(
            #             epoch, 
            #             num_samples=config.visualization.num_samples_per_save
            #         )

    # ... rest of existing code ...
"""


# ALTERNATIVE: Use the visualizer only in the test/evaluation method
# Add to the test() method after computing CER/WER:
"""
def test(self, epoch, tset='test'):
    # ... existing test code ...
    
    print('CER at epoch {}: {:.3f}'.format(epoch, cer_score))
    print('WER at epoch {}: {:.3f}'.format(epoch, wer_score))
    
    # Optional: Visualize a few samples with attention
    if self.visualizer is not None and tset == 'val':
        self.visualize_attention_samples(epoch, num_samples=3)
    
    self.net.train()
"""
