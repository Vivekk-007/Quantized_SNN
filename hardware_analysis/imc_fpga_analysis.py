def estimate_memory(bits, params):
    return bits * params / 8 / 1024  # KB

def estimate_mac(bits, macs):
    return macs * bits

params = 50*64 + 64*2
macs = params

for bits in [32, 8, 4, 1]:
    mem = estimate_memory(bits, params)
    mac = estimate_mac(bits, macs)
    print(f"{bits}-bit → Memory: {mem:.2f} KB, MAC cost: {mac}")
