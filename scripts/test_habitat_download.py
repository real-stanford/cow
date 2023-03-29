import habitat_sim
import habitat
print(f'habitat_sim.__version__: {habitat_sim.__version__}')
print(f'habitat.__version__: {habitat.__version__}')

assert habitat_sim.__version__ == '0.2.1'
assert habitat.__version__ == '0.2.1'

print('Looks good.')