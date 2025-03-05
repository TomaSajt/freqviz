{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = [
    pkgs.alsa-lib
    pkgs.pkg-config
    pkgs.cmake
    pkgs.xorg.libX11
    pkgs.xorg.libXrandr
    pkgs.xorg.libXinerama
    pkgs.xorg.libXcursor
    pkgs.xorg.libXi
    pkgs.libGL
    pkgs.rustPlatform.bindgenHook
  ];
  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.libGL ];
}
