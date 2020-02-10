import disentangled.dataset.shapes3d as shapes3d
import disentangled.visualize as vi

def main():
    dataset = shapes3d.load()
    vi.show(vi.stack(dataset, 5, 10))
    

if __name__ == '__main__':
    main()
