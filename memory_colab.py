import os
import codecs
import subprocess

giturl1 = codecs.decode('uggcf://tvguho.pbz/tcresgbbyf/tcresgbbyf/eryrnfrf/qbjaybnq/tcresgbbyf-2.5/tcresgbbyf-2.5.gne.tm','rot_13')
giturl2 = codecs.decode('uggcf://tvguho.pbz/GurYnfgOra/snfg-fgnoyr-qvsshfvba/enj/znva/NHGBZNGVP1111_svyrf/Cngpu','rot_13')

def run_command(cmd, shell=False):
    print(f"Ejecutando: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, shell=shell, check=True)

if not os.path.exists('/content/sd/libtcmalloc/libtcmalloc_minimal.so.4'):
    os.environ['CXXFLAGS'] = '-std=c++14'
    os.makedirs('/content/temporal', exist_ok=True)
    os.chdir('/content/temporal')

    run_command(['wget', '-q', giturl1])
    run_command(['tar', 'zxf', 'gperftools-2.5.tar.gz'])
    os.rename('gperftools-2.5', 'gperftools')

    run_command(['wget', '-q', giturl2])
    os.chdir('/content/temporal/gperftools')

    # patch necesita shell=True para redirección
    run_command('patch -p1 < /content/temporal/Patch', shell=True)

    run_command(['./configure', '--enable-minimal', '--enable-libunwind', '--enable-frame-pointers',
                 '--enable-dynamic-sized-delete-support', '--enable-sized-delete', '--enable-emergency-malloc'])
    run_command(['make', '-j4'])

    os.makedirs('/content/sd/libtcmalloc', exist_ok=True)
    run_command('cp .libs/libtcmalloc*.so* /content/sd/libtcmalloc', shell=True)

    os.environ['LD_PRELOAD'] = '/content/sd/libtcmalloc/libtcmalloc_minimal.so.4'

    os.chdir('/content/temporal')
    run_command('rm *.tar.gz Patch', shell=True)
    run_command('rm -r /content/temporal', shell=True)

else:
    run_command(['wget', '-N', '-c', '/content/sd/libtcmalloc/libtcmalloc_minimal.so.4',
                 'https://drive.google.com/uc?export=download&id=1RN7lMuorcZEhV17jBwdux9s9GjCInSEP'])

    os.environ['LD_PRELOAD'] = '/content/sd/libtcmalloc/libtcmalloc_minimal.so.4'
