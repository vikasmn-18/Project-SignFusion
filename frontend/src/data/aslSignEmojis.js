// ASL chart reference aligned to the user-provided mounted A-Z image.
// Images imported directly from assets so they resolve correctly with React/webpack.
import aImg from "../assets/a_.png";
import bImg from "../assets/b_.png";
import cImg from "../assets/c_.png";
import dImg from "../assets/d_.png";
import eImg from "../assets/e_.png";
import fImg from "../assets/f_.png";
import gImg from "../assets/g_.png";
import hImg from "../assets/h_.png";
import iImg from "../assets/i_.png";
import jImg from "../assets/j_.png";
import kImg from "../assets/k_.png";
import lImg from "../assets/l_.png";
import mImg from "../assets/m_.png";
import nImg from "../assets/n_.png";
import oImg from "../assets/o_.png";
import pImg from "../assets/p_.png";
import qImg from "../assets/q_.png";
import rImg from "../assets/r_.png";
import sImg from "../assets/s_.png";
import tImg from "../assets/t_.png";
import uImg from "../assets/u_.png";
import vImg from "../assets/v_.png";
import wImg from "../assets/w_.png";
import xImg from "../assets/x_.png";
import yImg from "../assets/y_.png";
import zImg from "../assets/z_.png";

export const ASL_SIGN_EMOJIS = [
  { letter: "A", symbol: "A", imagePath: aImg, cue: "Closed fist, thumb outside on the side." },
  { letter: "B", symbol: "B", imagePath: bImg, cue: "Palm open, four fingers straight together, thumb tucked." },
  { letter: "C", symbol: "C", imagePath: cImg, cue: "Hand curves into a C shape." },
  { letter: "D", symbol: "D", imagePath: dImg, cue: "Index finger up, thumb touches the middle finger." },
  { letter: "E", symbol: "E", imagePath: eImg, cue: "Fingers bend down over the thumb." },
  { letter: "F", symbol: "F", imagePath: fImg, cue: "Thumb and index make a circle, other fingers up." },
  { letter: "G", symbol: "G", imagePath: gImg, cue: "Index and thumb point sideways close together." },
  { letter: "H", symbol: "H", imagePath: hImg, cue: "Index and middle extend sideways together." },
  { letter: "I", symbol: "I", imagePath: iImg, cue: "Only the pinky finger is raised." },
  { letter: "J", symbol: "J", imagePath: jImg, cue: "Start from I and trace J downward in motion.", motion: true },
  { letter: "K", symbol: "K", imagePath: kImg, cue: "Index and middle up, thumb supports from between." },
  { letter: "L", symbol: "L", imagePath: lImg, cue: "Index up and thumb out form an L." },
  { letter: "M", symbol: "M", imagePath: mImg, cue: "Thumb tucks under three fingers." },
  { letter: "N", symbol: "N", imagePath: nImg, cue: "Thumb tucks under two fingers." },
  { letter: "O", symbol: "O", imagePath: oImg, cue: "Fingers and thumb curve into an O." },
  { letter: "P", symbol: "P", imagePath: pImg, cue: "K handshape angled downward." },
  { letter: "Q", symbol: "Q", imagePath: qImg, cue: "G handshape angled downward." },
  { letter: "R", symbol: "R", imagePath: rImg, cue: "Index and middle fingers crossed." },
  { letter: "S", symbol: "S", imagePath: sImg, cue: "Closed fist with thumb across the front." },
  { letter: "T", symbol: "T", imagePath: tImg, cue: "Thumb tucked between index and middle." },
  { letter: "U", symbol: "U", imagePath: uImg, cue: "Index and middle fingers up together." },
  { letter: "V", symbol: "V", imagePath: vImg, cue: "Index and middle fingers spread into a V." },
  { letter: "W", symbol: "W", imagePath: wImg, cue: "Three fingers up: index, middle, and ring." },
  { letter: "X", symbol: "X", imagePath: xImg, cue: "Index finger bent into a hook." },
  { letter: "Y", symbol: "Y", imagePath: yImg, cue: "Thumb and pinky out, other fingers closed." },
  { letter: "Z", symbol: "Z", imagePath: zImg, cue: "Index finger traces Z in motion.", motion: true },
];

export const ASL_SIGN_EMOJI_MAP = Object.fromEntries(
  ASL_SIGN_EMOJIS.map((item) => [item.letter, item])
);
