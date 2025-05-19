#set text(size: 14pt)
#set page(paper: "a4")
#show heading: it => block(below: 1em, it)

#align(center)[
  #image("assets/Unibersidad_ng_Pilipinas_Mindanao.png", width: 30%)

  #block(above: 5em, below: 8em)[
    = Facial Landmark Visualizer
  ]

  Lada, Legolas Tyrael B.

  #v(4em)

  #block(width: 75%)[
    #set par(leading: 1.6em)
    in fulfillment of the requirements of #box([CMSC 191 -- COMPUTER VISION IN PYTHON])
  ]

  #v(6em)

  Submitted to

  Assoc. Prof. Armacheska R. Mesa-Satina, Ph.D.
]

#align(center + bottom)[
  May 2025
]

#pagebreak()
#set par(justify: true, first-line-indent: (amount: 2em, all: true), leading: 1.3em)
#set page(margin: (left: 1.25in))

= Abstract

#block(width: 95%)[
  #align(left)[
    Arguably one of the harder parts of image processing and computer vision
    is visualizing the preprocessing steps required to get to workable data,
    that is, data that is easy to analyze and perform logic on. Moreover,
    visualizing these preprocessing steps can be cumbersome, especially for
    steps that are more complex than applying simple filters and
    transformations over the image. This project acts as a tool to be able to
    visualize steps in isolation or in combination with the other
    preprocessing steps required in obtaining facial landmarks via _Haar
    cascades_, a simplified approach to facial detection.
  ]
]

= Objectives

This project aims to provide an easier way to visualize the preprocessing steps
involved in obtaining facial recognition data from a Haar cascade produced by
`dlib`. Specifically,

#block(inset: (left: 0.5in))[
  + Allows an intuitive interface for activating showing specific preprocessing
    steps;
  + allows for overlap of displayed preprocessing steps;
  + apply the preprocessing to a live webcam feed;
]

= Project Features

The project allows for the toggling of four preprocessing steps in particular.
These are: locating and predicting _68_ (obtained using
`shape_predictor_68_face_landmarks.dat` and `dlib`) facial landmark points,
displaying the convex hull formed by the landmark points, displaying the mask
formed by the convex hull, and the _Delauney triangles_ formed by the landmark
points.

These are all laid out using _Streamlit._ The aforementioned settings are then
applied to the live webcam feed.

= Techniques Used

This project utilizes a pre-trained model with `dlib`, as done in one of the
laboratory works. It also uses various _OpenCV_ functions to perform
preprocessing and filters on the image, besides loading the image from the
webcam. The image was turned grayscale, blurred, and then morphed using the
_open_ morphological transform, to improve detection of the face.

= Screenshots and Workflow

#{
  set par(leading: 0.65em)

  figure(image("./assets/sc1.png"), caption: [A live feed without any filters visualized.])

  figure(image("./assets/sc2.png"), caption: [A live feed visualizing the landmark points and convex hull.])

  figure(
    image("./assets/sc3.png"),
    caption: [A live feed visualizing the Delauney triangles formed by the landmark points.],
  )

  figure(
    image("./assets/sc4.png"),
    caption: [A live feed visualizing the image mask formed by the convex hull along with its landmark points.],
  )
}

= Challenges and Limitations

The main challenge that plagued me for the development process of this project
was the ideation process. I must have spent about 3 days trying to think of a
project idea. Additionally, it is also my first time using Streamlit, so I
encountered issues regarding its state handling, as it refreshes the page
on input.

I also had to relearn a bit of the preprocessing done in using a Haar cascade
model, particularly all of the steps being visualized in the project.

Lastly, I tried to let the settings be toggleable without closing the live
feed, but I couldn't figure it out, because of the inexperience with Streamlit.

= Future Enhancements

I believe the user experience can be improved by being able to toggle the
filters while the live feed is active. And being able to customize the
appearance of the filters would also be good.

And of course, adding more steps to be visualized is also a great way of
improving upon this project.

= Reflection

This project was very helpful in helping cement my learnings in the course,
as it had me culminate them into a project that I chose. There was definitely
a struggle in the ideation but only because of the fact that computer vision
lends itself to an overwhelming amount of possible projects, which was very
fun to think about, in hindsight. I wish I could have been able to do a whole
lot more, but there was a part of me that just felt like over-investing into
my first proper computer vision project would give me very diminishing returns.
Although I am not fully satisfied with it, I still am very much looking forward
to applying more of what I have learned in the future, and I think this project
has been a very good springboard, in that regard.

= References

The code was written entirely by me, besides the dataset, which was obtained
from Laboratory Work \#7 of this class.
