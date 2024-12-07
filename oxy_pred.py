
#%%
model = UNet()
model.load_state_dict(torch.load("model_weights_4E.pth"))


#%%
# Effectuer des prédictions pour toutes les images de test
with torch.no_grad():  # Désactiver la rétropropagation pour gagner en mémoire
    for i, (inputs, _) in enumerate(train_loader):
        # Passer les images dans le modèle
        outputs = model(inputs)

        # Afficher l'image originale et la reconstruction
        if i == 0:  # Afficher la première image pour exemple
            input_img = inputs[0].cpu().numpy().transpose(
                1, 2, 0)  # Convertir en format (H, W, C)
            output_img = outputs[0].cpu().numpy().transpose(1, 2, 0)
            gt = _[0].cpu().numpy().transpose(1, 2, 0)

            plt.figure(figsize=(16, 4))

            # Afficher l'image d'entrée
            plt.subplot(1, 4, 1)
            # Utiliser .squeeze() si l'image est en niveaux de gris
            plt.imshow(input_img.squeeze())
            plt.title("Image d'entrée")

            # Afficher la reconstruction
            plt.subplot(1, 4, 2)
            plt.imshow(output_img.squeeze())
            plt.title("Reconstruction")

            # Afficher la Ground truth
            plt.subplot(1, 4, 3)
            plt.imshow(gt.squeeze())
            plt.title("Ground Truth")

            # Afficher la Ground truth
            plt.subplot(1, 4, 4)
            plt.imshow(np.sqrt((output_img.squeeze() - gt.squeeze())**2))
            plt.title("MSE")

            plt.show()
            break
#%%
# Générer des images à partir de bruit aléatoire
with torch.no_grad():  # Désactiver la rétropropagation pour gagner en mémoire

    # Par exemple, bruit aléatoire de taille 28x28
    noise = torch.rand(1, 1, 256, 256)
    # Utiliser le décodeur du modèle pour générer une image
    generated_image = model(noise)

    # Afficher l'image générée
    generated_image = generated_image.squeeze().cpu(
    ).numpy()  # Enlever la dimension inutile

    plt.subplot(1, 2, 1)
    # Utiliser .squeeze() si l'image est en niveaux de gris
    plt.imshow(noise.squeeze().cpu().numpy())
    plt.title("Image d'entrée")
    plt.subplot(1, 2, 2)
    plt.imshow(generated_image)
    plt.title("Image générée")
    plt.show()
