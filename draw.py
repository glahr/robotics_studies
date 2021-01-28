import pygame

def reference():

    pygame.init()

    #logo = pygame.image.load("logo32x32.png")
    #pygame.display.set_icon(logo)
    image = pygame.image.load("Drawing.png")
    pygame.display.set_caption("Reference image")

    screen = pygame.display.set_mode((450,450))
    screen.blit(image, (0,0))

    pygame.display.flip()

    running = True

    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False


if __name__ == "__main__":
    reference()